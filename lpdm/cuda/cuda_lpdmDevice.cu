/* FastEddy(R) fork: GPU-resident backward LPDM -- device implementation.
 *
 * A LINE-BY-LINE TRANSLITERATION of lpdm/model.py (_local and _run_one) and the parts of
 * lpdm/fields.py they call. Read the two together: where this file departs from the
 * Python in anything but syntax, the departure is a bug, and bin/test_gpu_lpdm.py is what
 * finds it. The comments here say WHICH Python line each block is, so the two can be
 * diffed by eye.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "cuda_lpdmDevice_cu.h"

#define CE_SGS   0.93f      /* FastEddy's c_e, cuda_sgstkeDevice.cu; lpdm/fields.py C_E   */
#define GRAV_    9.81f
#define TWO3     (2.0/3.0)

#define CK(call) do{ cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"LPDM CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
  return -1; } }while(0)

/* ---------------------------------------------------------------- device state ------ */
static LpdmGrid  G;
static int       NXP, NYP;        /* wrap-padded horizontal extents: nx+1, ny+1          */
static size_t    FLDSZ;           /* nz*NYP*NXP                                          */
static __half   *fld_d = NULL;    /* [LPDM_NFLD][nt][nz][NYP][NXP]                       */
static float    *ust_d = NULL;    /* [nt][NYP][NXP]                                      */
static float    *z0m_d = NULL;
static double   *tstamp_d = NULL; /* [nt] model time of each ring slot                   */
static double   *tstamp_h = NULL;
static float    *zk_d = NULL, *zg_d = NULL, *sx_d = NULL, *sy_d = NULL, *dm_d = NULL;
static int       have_terrain = 0, have_dmap = 0;
static double   *offz_d = NULL, *offf_d = NULL, *offs_d = NULL;  /* floor table + slopes */
static int       n_off = 0;
/* DELIVERY OF THE FLOOR. Production is MULTIPLICATIVE -- sigma^2 = sc(z) (2/3)e -- and
 * additive (sigma^2 = (2/3)e + delta(z)) is a recorded negative result kept so the two can
 * be measured on the same fields (SIXTH_PASS_RESULTS.md). Both are here because the CPU
 * path has both, and a port that silently supported only one would be a different model
 * from the one it is being scored against. */
static int       floor_mode = 0;   /* 0 none, 1 multiplicative, 2 additive */
static unsigned char *cover_d = NULL;
static int       n_cover = 0;
static double   *cov_num_d = NULL, *cov_den_d = NULL;
static double   *flux_d = NULL, *conc_d = NULL;
static double   *sums_d = NULL;   /* [0]=sum_flux_all [1]=sum_conc_all                   */
static long long *cnt_d = NULL;   /* [0]=n_particles [1]=n_touchdown [2]=td_kept [3]=ovf */
static float    *td_dx_d=NULL,*td_dy_d=NULL,*td_wt_d=NULL,*td_ag_d=NULL;
static long long TD_CAP = 0;
static float    *scratch_d = NULL;   /* staging for one interior field, fp32             */
static float    *scratch2_d = NULL;
static int       NPMAX = 0;

/* ---------------------------------------------------------------- small helpers ----- */
__device__ __forceinline__ double dwrap(double v, double n){
  double r = fmod(v, n); return r < 0.0 ? r + n : r;
}
/* lpdm/fields.py::_corner -- clamped base index and weight, mode='nearest'. */
__device__ __forceinline__ int corner(double c, int n, double *w){
  if(c < 0.0) c = 0.0; if(c > (double)(n-1)) c = (double)(n-1);
  int i0 = (int)floor(c); if(i0 > n-2) i0 = (n-2 > 0 ? n-2 : 0); if(i0 < 0) i0 = 0;
  *w = c - (double)i0; return i0;
}
/* np.interp on a monotone table, with CONSTANT extrapolation at both ends. */
__device__ __forceinline__ double tinterp(const double *xs, const double *ys, int n, double x){
  if(n <= 0) return 0.0;
  if(x <= xs[0]) return ys[0];
  if(x >= xs[n-1]) return ys[n-1];
  int lo = 0, hi = n-1;
  while(hi - lo > 1){ int m = (lo+hi)>>1; if(xs[m] <= x) lo = m; else hi = m; }
  double f = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
__device__ __forceinline__ int tinterval(const double *xs, int n, double x){
  if(x < xs[0] || x > xs[n-1]) return -1;
  int lo = 0, hi = n-1;
  while(hi - lo > 1){ int m = (lo+hi)>>1; if(xs[m] <= x) lo = m; else hi = m; }
  return lo;
}

/* ---------------------------------------------------------------- ingest kernels ---- */
/* Wrap-pad one interior (nz,ny,nx) fp32 field into the (nz,NYP,NXP) fp16 ring slot.
 * lpdm/fields.py::_pad -- the pad is what lets linear interpolation cross the periodic
 * seam without a modulo in the integrator's inner loop. */
__global__ void kPadTo16(const float *src, __half *dst, int nx, int ny, int nz,
                         int nxp, int nyp){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int n = nz*nyp*nxp; if(idx >= n) return;
  int i = idx % nxp, j = (idx/nxp) % nyp, k = idx/(nxp*nyp);
  int si = (i == nx) ? 0 : i, sj = (j == ny) ? 0 : j;
  dst[idx] = __float2half(src[(size_t)k*ny*nx + (size_t)sj*nx + si]);
}
__global__ void kPadTo32(const float *src, float *dst, int nx, int ny, int nxp, int nyp){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int n = nyp*nxp; if(idx >= n) return;
  int i = idx % nxp, j = idx / nxp;
  int si = (i == nx) ? 0 : i, sj = (j == ny) ? 0 : j;
  dst[idx] = src[(size_t)sj*nx + si];
}
/* eps and dsig2dz, EXACTLY as lpdm/fields.py derives them:
 *     dthdz = np.gradient(theta, zk, axis=0)         (non-uniform second-order)
 *     n2    = (g/theta) dthdz
 *     len1  = 0.76 sqrt(e)/sqrt(max(n2,1e-12))
 *     ell   = where(n2>0, max(min(len1, Delta), 1e-2), Delta)
 *     eps   = C_E e^{3/2} / max(ell, 1e-2),  then clipped at 1e-6 for the fp16 cache
 *     ds2   = np.gradient((2/3) e, zk, axis=0)
 * np.gradient on a non-uniform axis uses the second-order formula with unequal spacings
 * in the interior and a one-sided second-order formula at the two ends; both are written
 * out here rather than approximated, because the drift term IS this derivative. */
__global__ void kDerive(const float *e, const float *th, const float *zk,
                        __half *eps_out, __half *ds2_out,
                        int nx, int ny, int nz, int nxp, int nyp, float dx, float dy){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int ncol = ny*nx; if(idx >= ncol) return;
  int i = idx % nx, j = idx / nx;
  for(int k = 0; k < nz; ++k){
    /* non-uniform np.gradient along k */
    double dthdz, ds2;
    double s2m, s2c, s2p;
    double thm, thc, thp, hm, hp;
    thc = th[(size_t)k*ncol + idx];
    s2c = TWO3 * (double)e[(size_t)k*ncol + idx];
    if(nz < 2){ dthdz = 0.0; ds2 = 0.0; }
    else if(k == 0){
      thp = th[(size_t)1*ncol + idx];
      double th2 = th[(size_t)(nz>2?2:1)*ncol + idx];
      s2p = TWO3 * (double)e[(size_t)1*ncol + idx];
      double s22 = TWO3 * (double)e[(size_t)(nz>2?2:1)*ncol + idx];
      if(nz > 2){
        double h0 = zk[1]-zk[0], h1 = zk[2]-zk[1];
        double a = -(2.0*h0 + h1)/(h0*(h0+h1)), b = (h0+h1)/(h0*h1), c = -h0/(h1*(h0+h1));
        dthdz = a*thc + b*thp + c*th2;   ds2 = a*s2c + b*s2p + c*s22;
      }else{ dthdz = (thp-thc)/(zk[1]-zk[0]); ds2 = (s2p-s2c)/(zk[1]-zk[0]); }
    }else if(k == nz-1){
      thm = th[(size_t)(nz-2)*ncol + idx];
      double th2 = th[(size_t)(nz>2?nz-3:nz-2)*ncol + idx];
      s2m = TWO3 * (double)e[(size_t)(nz-2)*ncol + idx];
      double s22 = TWO3 * (double)e[(size_t)(nz>2?nz-3:nz-2)*ncol + idx];
      if(nz > 2){
        double h0 = zk[nz-2]-zk[nz-3], h1 = zk[nz-1]-zk[nz-2];
        double a = h1/(h0*(h0+h1)), b = -(h0+h1)/(h0*h1), c = (h0+2.0*h1)/(h1*(h0+h1));
        dthdz = a*th2 + b*thm + c*thc;   ds2 = a*s22 + b*s2m + c*s2c;
      }else{ dthdz = (thc-thm)/(zk[nz-1]-zk[nz-2]); ds2 = (s2c-s2m)/(zk[nz-1]-zk[nz-2]); }
    }else{
      thm = th[(size_t)(k-1)*ncol + idx]; thp = th[(size_t)(k+1)*ncol + idx];
      s2m = TWO3 * (double)e[(size_t)(k-1)*ncol + idx];
      s2p = TWO3 * (double)e[(size_t)(k+1)*ncol + idx];
      hm = zk[k]-zk[k-1]; hp = zk[k+1]-zk[k];
      double a = -hp/(hm*(hm+hp)), b = (hp-hm)/(hm*hp), c = hm/(hp*(hm+hp));
      dthdz = a*thm + b*thc + c*thp;   ds2 = a*s2m + b*s2c + c*s2p;
    }
    /* Delta from the flat-column dz, matching lpdm/fields.py's np.gradient(zk) */
    double dzc;
    if(nz < 2) dzc = 1.0;
    else if(k == 0) dzc = zk[1]-zk[0];
    else if(k == nz-1) dzc = zk[nz-1]-zk[nz-2];
    else dzc = 0.5*(zk[k+1]-zk[k-1]);
    double delta = cbrt((double)dx * (double)dy * dzc);
    double ed = (double)e[(size_t)k*ncol + idx]; if(ed < 0.0) ed = 0.0;
    double n2 = (GRAV_ / thc) * dthdz;
    double ell;
    if(n2 > 0.0){
      double len1 = 0.76 * sqrt(ed) / sqrt(n2 > 1e-12 ? n2 : 1e-12);
      ell = len1 < delta ? len1 : delta; if(ell < 1e-2) ell = 1e-2;
    } else ell = delta;
    double epsv = CE_SGS * pow(ed, 1.5) / (ell > 1e-2 ? ell : 1e-2);
    if(epsv < 1e-6) epsv = 1e-6;          /* fp16 subnormals bottom out near 6e-8 */
    /* write into the padded slot */
    size_t o = (size_t)k*nyp*nxp + (size_t)j*nxp + i;
    eps_out[o] = __float2half((float)epsv);
    ds2_out[o] = __float2half((float)ds2);
    if(i == 0){ size_t oe = (size_t)k*nyp*nxp + (size_t)j*nxp + nx;
                eps_out[oe] = eps_out[o]; ds2_out[oe] = ds2_out[o]; }
    if(j == 0){ size_t oe = (size_t)k*nyp*nxp + (size_t)ny*nxp + i;
                eps_out[oe] = eps_out[o]; ds2_out[oe] = ds2_out[o]; }
    if(i == 0 && j == 0){ size_t oe = (size_t)k*nyp*nxp + (size_t)ny*nxp + nx;
                eps_out[oe] = eps_out[o]; ds2_out[oe] = ds2_out[o]; }
  }
}

/* ---------------------------------------------------------------- device sampling --- */
struct Ctx {
  int nx, ny, nz, nt, nxp, nyp;
  double dx, dy, x0, y0, zC, dt_dump, t0_ring;
  const __half *fld; const float *ust; const float *z0m;
  const float *zk; const float *zg; const float *sx; const float *sy; const float *dm;
  int have_terrain, have_dmap;
  const double *offz, *offf, *offs; int n_off; int floor_mode;
  double z_ref, z_top;
};

__device__ __forceinline__ double bilin2(const float *a, int nxp, int nyp,
                                         double fj, double fi){
  double wj, wi; int j0 = corner(fj, nyp, &wj), i0 = corner(fi, nxp, &wi);
  int j1 = j0+1 < nyp ? j0+1 : nyp-1, i1 = i0+1 < nxp ? i0+1 : nxp-1;
  return (1-wj)*((1-wi)*a[j0*nxp+i0] + wi*a[j0*nxp+i1])
       +    wj *((1-wi)*a[j1*nxp+i0] + wi*a[j1*nxp+i1]);
}
/* zg / slope / dmap live on the UNPADDED (ny,nx) grid and are sampled with grid-wrap,
 * matching scipy.ndimage.map_coordinates(mode='grid-wrap') in lpdm/fields.py::ground. */
__device__ __forceinline__ double bilinWrap(const float *a, int nx, int ny,
                                            double fj, double fi){
  double yj = dwrap(fj, (double)ny), xi = dwrap(fi, (double)nx);
  int j0 = (int)floor(yj), i0 = (int)floor(xi);
  double wj = yj - j0, wi = xi - i0;
  int j1 = (j0+1) % ny, i1 = (i0+1) % nx;
  j0 %= ny; i0 %= nx;
  return (1-wj)*((1-wi)*a[j0*nx+i0] + wi*a[j0*nx+i1])
       +    wj *((1-wi)*a[j1*nx+i0] + wi*a[j1*nx+i1]);
}

__device__ void sample4(const Ctx &c, int fldbase_ok, double ft, double fk,
                        double fj, double fi, double *out /*[LPDM_NFLD]*/){
  double wt_, wk_, wj_, wi_;
  int t0 = corner(ft, c.nt, &wt_), k0 = corner(fk, c.nz, &wk_);
  int j0 = corner(fj, c.nyp, &wj_), i0 = corner(fi, c.nxp, &wi_);
  int t1 = t0+1 < c.nt ? t0+1 : c.nt-1, k1 = k0+1 < c.nz ? k0+1 : c.nz-1;
  int j1 = j0+1 < c.nyp ? j0+1 : c.nyp-1, i1 = i0+1 < c.nxp ? i0+1 : c.nxp-1;
  const size_t slice = (size_t)c.nz*c.nyp*c.nxp;
  int tt[2] = {t0,t1}; double ct[2] = {1.0-wt_, wt_};
  int kk[2] = {k0,k1}; double ck[2] = {1.0-wk_, wk_};
  int jj[2] = {j0,j1}; double cj[2] = {1.0-wj_, wj_};
  int ii[2] = {i0,i1}; double ci[2] = {1.0-wi_, wi_};
  for(int f = 0; f < LPDM_NFLD; ++f) out[f] = 0.0;
  for(int a = 0; a < 2; ++a) for(int b = 0; b < 2; ++b)
  for(int d = 0; d < 2; ++d) for(int e = 0; e < 2; ++e){
    double w = ct[a]*ck[b]*cj[d]*ci[e]; if(w == 0.0) continue;
    size_t base = (size_t)tt[a]*slice + (size_t)kk[b]*c.nyp*c.nxp
                + (size_t)jj[d]*c.nxp + ii[e];
    for(int f = 0; f < LPDM_NFLD; ++f)
      out[f] += w * (double)__half2float(c.fld[(size_t)f*c.nt*slice + base]);
  }
}
__device__ void sample2(const Ctx &c, double ft, double fj, double fi,
                        double *ustar, double *z0m){
  double wt_, wj_, wi_;
  int t0 = corner(ft, c.nt, &wt_), j0 = corner(fj, c.nyp, &wj_), i0 = corner(fi, c.nxp, &wi_);
  int t1 = t0+1 < c.nt ? t0+1 : c.nt-1;
  int j1 = j0+1 < c.nyp ? j0+1 : c.nyp-1, i1 = i0+1 < c.nxp ? i0+1 : c.nxp-1;
  const size_t sl = (size_t)c.nyp*c.nxp;
  double us = 0.0, z0 = 0.0;
  int tt[2] = {t0,t1}; double ct[2] = {1.0-wt_, wt_};
  int jj[2] = {j0,j1}; double cj[2] = {1.0-wj_, wj_};
  int ii[2] = {i0,i1}; double ci[2] = {1.0-wi_, wi_};
  for(int a = 0; a < 2; ++a) for(int d = 0; d < 2; ++d) for(int e = 0; e < 2; ++e){
    double w = ct[a]*cj[d]*ci[e]; if(w == 0.0) continue;
    size_t o = (size_t)tt[a]*sl + (size_t)jj[d]*c.nxp + ii[e];
    us += w * c.ust[o]; z0 += w * c.z0m[o];
  }
  *ustar = us; *z0m = z0;
}

/* lpdm/fields.py::kindex -- fractional k from physical height, undoing the stretch. */
__device__ __forceinline__ double kindex(const Ctx &c, double z, double zg){
  double F = (z - zg) * c.zC / fmax(c.zC - zg, 1e-12);
  /* np.interp(F, Fk, arange(nz)) with constant extrapolation */
  if(F <= c.zk[0]) return 0.0;
  if(F >= c.zk[c.nz-1]) return (double)(c.nz-1);
  int lo = 0, hi = c.nz-1;
  while(hi - lo > 1){ int m = (lo+hi)>>1; if((double)c.zk[m] <= F) lo = m; else hi = m; }
  return (double)lo + (F - c.zk[lo]) / ((double)c.zk[hi] - (double)c.zk[lo]);
}

/* lpdm/model.py::_local. Signature mirrors the Python return exactly. */
__device__ void localStats(const Ctx &c, double x, double y, double z, double t,
                           double *U, double *V, double *W, double *sig2,
                           double *eps, double *ds2z, double *ustar_out,
                           double *zg_out, double *zagl_out,
                           double z_touch){
  double fi = dwrap((x - c.x0)/c.dx, (double)c.nx);
  double fj = dwrap((y - c.y0)/c.dy, (double)c.ny);
  double zg = c.have_terrain ? bilinWrap(c.zg, c.nx, c.ny, fj, fi) : 0.0;
  double zagl = z - zg;
  double lo = z_touch*0.5, hi = c.z_top - 1.0;
  if(zagl < lo) zagl = lo; if(zagl > hi) zagl = hi;
  double zq = zagl > c.z_ref ? zagl : c.z_ref;
  double fk = kindex(c, zg + zq, zg);
  double ft = (t - c.t0_ring)/fmax(c.dt_dump, 1e-12);
  if(ft < 0.0) ft = 0.0; if(ft > (double)(c.nt-1)) ft = (double)(c.nt-1);
  double f[LPDM_NFLD];
  sample4(c, 1, ft, fk, fj, fi, f);
  double u = f[LPDM_FLD_U], v = f[LPDM_FLD_V], w = f[LPDM_FLD_W];
  double e = f[LPDM_FLD_E], ep = f[LPDM_FLD_EPS], ds = f[LPDM_FLD_DS2];
  double us, z0; sample2(c, ft, fj, fi, &us, &z0);

  int below = (zagl < c.z_ref);
  if(below){
    double z0b = z0 > 1e-4 ? z0 : 1e-4;
    double db = c.have_dmap ? bilinWrap(c.dm, c.nx, c.ny, fj, fi) : 0.0;
    double zeff = zagl - db; double fl = 1.001*z0b; if(zeff < fl) zeff = fl;
    double zref_eff = c.z_ref - db; double fl2 = 1.002*z0b;
    if(zref_eff < fl2) zref_eff = fl2;
    double scal = log(zeff/z0b) / log(zref_eff/z0b);
    if(scal < 0.0) scal = 0.0; if(scal > 1.0) scal = 1.0;
    u *= scal; v *= scal; w *= zagl / c.z_ref;
    ep *= c.z_ref / zagl;
    ds = 0.0;
  }
  /* lpdm/model.py::_local, the floor block. MULTIPLICATIVE is production. */
  double sig2_raw = TWO3 * e;
  double s2 = sig2_raw;
  if(c.n_off > 0 && c.floor_mode == 1){
    /* sc(z) sampled at zagl (NOT clamped to z_ref -- the Python samples the raw zagl
     * here), and its derivative is the PIECEWISE-CONSTANT slope of the interpolant the
     * model actually transports, not a smoothed central difference. */
    double sc = tinterp(c.offz, c.offf, c.n_off, zagl);
    int iv = tinterval(c.offz, c.n_off, zagl);
    double dscdz = (iv < 0) ? 0.0 : c.offs[iv];
    ds = sc*ds + TWO3*e*dscdz;          /* product rule; sc == 1 reduces to the field */
    if(below) ds = 0.0;
    s2 = sc*sig2_raw;
  } else if(c.n_off > 0 && c.floor_mode == 2){
    double zqf = zagl > c.z_ref ? zagl : c.z_ref;
    s2 += tinterp(c.offz, c.offf, c.n_off, zqf);
    int iv = tinterval(c.offz, c.n_off, zqf);
    double doff = (iv < 0) ? 0.0 : c.offs[iv];
    if(below) doff = 0.0;
    ds += doff;
  }
  if(s2 < 1e-6) s2 = 1e-6;
  /* eps consistency, with BOTH sides floored identically (FASTEDDY_TRAPS.md 11) */
  double denom = sig2_raw > 1e-6 ? sig2_raw : 1e-6;
  double ratio = s2/denom; if(ratio < 1.0) ratio = 1.0;
  ep *= ratio;
  if(ep < 1e-8) ep = 1e-8;
  *U = u; *V = v; *W = w; *sig2 = s2; *eps = ep; *ds2z = ds; *ustar_out = us;
  *zg_out = zg; *zagl_out = zagl;
}

/* ---------------------------------------------------------------- the integrator ---- */
__global__ void kIntegrate(Ctx c, LpdmOpts o, int n,
                           const double *x_rel_a, const double *y_rel_a,
                           const double *z_rel_a, const double *t_rel,
                           double x_ref, double y_ref, double i_r, double j_r,
                           double *fin_x, double *fin_y, double *fin_z,
                           double *flux, double *conc, double *sums,
                           long long *cnt, const unsigned char *cover, int n_cover,
                           double *cov_num, double *cov_den,
                           float *td_dx, float *td_dy, float *td_wt, float *td_ag,
                           long long td_cap){
  int p = blockIdx.x*blockDim.x + threadIdx.x; if(p >= n) return;
  curandStatePhilox4_32_10_t rs;
  curand_init(o.seed, (unsigned long long)p, 0ULL, &rs);

  double x_rel = x_rel_a[p], y_rel = y_rel_a[p];
  double x = x_rel, y = y_rel, z = z_rel_a[p], t = t_rel[p], elapsed = 0.0;
  double x0r = x, y0r = y;
  double U,V,W,sig2,eps,ds2z,ustar,zg,zagl;
  localStats(c, x, y, z, t, &U,&V,&W,&sig2,&eps,&ds2z,&ustar,&zg,&zagl, o.z_touch);
  double sg = sqrt(sig2);
  double us0 = curand_normal_double(&rs)*sg;
  double us1 = curand_normal_double(&rs)*sg;
  double us2 = curand_normal_double(&rs)*sg;
  double w_release = 0.0;           /* streamline-frame w at release; set at it == 0     */
  double sgn = (double)o.direction;
  int first = 1;

  for(int it = 0; it < o.max_iter; ++it){
    localStats(c, x, y, z, t, &U,&V,&W,&sig2,&eps,&ds2z,&ustar,&zg,&zagl, o.z_touch);
    double sig = sqrt(sig2);
    double TL = 2.0*sig2/(o.c0*eps);
    double dt = o.dt_frac*TL;
    if(dt < o.dt_min) dt = o.dt_min; if(dt > o.dt_max) dt = o.dt_max;
    double rem = o.t_limit - elapsed; if(rem < 0.0) rem = 0.0;
    if(dt > rem) dt = rem;

    if(first){
      /* lpdm/driver.py::streamline_w on the release velocity, all three components. */
      double ru = U + us0, rv = V + us1, rw = W + us2;
      w_release = rw*cos(o.pitch)
                - (ru*cos(o.yaw) + rv*sin(o.yaw))*sin(o.pitch);
      first = 0;
    }

    double a_ = exp(-dt/TL);
    double b_ = sig*sqrt(fmax(1.0 - a_*a_, 0.0));
    us0 = us0*a_ + b_*curand_normal_double(&rs);
    us1 = us1*a_ + b_*curand_normal_double(&rs);
    us2 = us2*a_ + b_*curand_normal_double(&rs);

    /* Thomson reverse-time sigma^2-gradient drift, isotropic split (aniso == 1). */
    double drift_w = 0.5*ds2z*(1.0 + us2*us2/sig2);
    double drift_u = 0.5*ds2z*(us0*us2/sig2);
    double drift_v = 0.5*ds2z*(us1*us2/sig2);
    us0 += sgn*drift_u*dt; us1 += sgn*drift_v*dt; us2 += sgn*drift_w*dt;
    double cap = 5.0*sig;
    if(us0 >  cap) us0 =  cap; if(us0 < -cap) us0 = -cap;
    if(us1 >  cap) us1 =  cap; if(us1 < -cap) us1 = -cap;
    if(us2 >  cap) us2 =  cap; if(us2 < -cap) us2 = -cap;

    double wtot = W + us2, utot = U + us0, vtot = V + us1;
    double fi = dwrap((x - c.x0)/c.dx, (double)c.nx);
    double fj = dwrap((y - c.y0)/c.dy, (double)c.ny);
    double slx = c.have_terrain ? bilinWrap(c.sx, c.nx, c.ny, fj, fi) : 0.0;
    double sly = c.have_terrain ? bilinWrap(c.sy, c.nx, c.ny, fj, fi) : 0.0;
    double w_agl = wtot - utot*slx - vtot*sly;
    x += sgn*utot*dt; y += sgn*vtot*dt;
    double znew = z + sgn*wtot*dt;
    t += sgn*dt; elapsed += dt;

    double fi2 = dwrap((x - c.x0)/c.dx, (double)c.nx);
    double fj2 = dwrap((y - c.y0)/c.dy, (double)c.ny);
    double zg2 = c.have_terrain ? bilinWrap(c.zg, c.nx, c.ny, fj2, fi2) : 0.0;
    int hit = (znew - zg2) < o.z_touch;
    if(hit){
      /* ---- deposit this touchdown -------------------------------------------------- */
      double aw = fabs(w_agl); if(aw < o.w_floor) aw = o.w_floor;
      double wt_c = 2.0/aw;
      double wt_f = w_release*wt_c;
      atomicAdd(&sums[0], wt_f); atomicAdd(&sums[1], wt_c);
      atomicAdd((unsigned long long*)&cnt[1], 1ULL);
      /* STATIC frame, folded by LES index -- lpdm/driver.py, so the raster cell IS the
       * LES column and the cover attribution folds identically. */
      double Xm = (dwrap(fi2 + 0.5, (double)c.nx) - 0.5 - i_r) * c.dx;
      double Ym = (dwrap(fj2 + 0.5, (double)c.ny) - 0.5 - j_r) * c.dy;
      /* cloud-in-cell on the raster whose cells are the LES columns */
      double uu = Xm/c.dx + i_r, vv = Ym/c.dy + j_r;
      int ci0 = (int)floor(uu), cj0 = (int)floor(vv);
      double fx = uu - ci0, fy = vv - cj0;
      for(int dj = 0; dj < 2; ++dj) for(int di = 0; di < 2; ++di){
        double ww = (di ? fx : 1.0-fx) * (dj ? fy : 1.0-fy);
        int ii = ((ci0+di) % c.nx + c.nx) % c.nx;
        int jj = ((cj0+dj) % c.ny + c.ny) % c.ny;
        atomicAdd(&flux[jj*c.nx+ii], wt_f*ww);
        atomicAdd(&conc[jj*c.nx+ii], wt_c*ww);
      }
      /* land-cover attribution at NEAREST grid point in LES index space */
      if(n_cover > 0){
        int ii = (int)(dwrap(floor(fi2+0.5), (double)c.nx));
        int jj = (int)(dwrap(floor(fj2+0.5), (double)c.ny));
        atomicAdd(&cov_den[0], wt_f);
        for(int m = 0; m < n_cover; ++m)
          if(cover[(size_t)m*c.ny*c.nx + (size_t)jj*c.nx + ii])
            atomicAdd(&cov_num[m], wt_f);
      }
      /* bounded uniform (Bernoulli) subsample of the raw touchdowns, UNFOLDED */
      if(td_cap > 0 && curand_uniform_double(&rs) < o.td_prob){
        long long s = atomicAdd((unsigned long long*)&cnt[2], 1ULL);
        if(s < td_cap){
          td_dx[s] = (float)(x - x_ref); td_dy[s] = (float)(y - y_ref);
          td_wt[s] = (float)wt_f;        td_ag[s] = (float)elapsed;
        } else atomicExch((unsigned long long*)&cnt[3], 1ULL);
      }
      if(o.reflect){ znew = 2.0*(zg2 + o.z_touch) - znew; us2 = -us2; }
      else znew = zg2 + o.z_touch;
    }
    if(znew > c.z_top){ znew = 2.0*c.z_top - znew; us2 = -us2; }
    z = znew;

    double ddx = x - x0r, ddy = y - y0r;
    double disp = sqrt(ddx*ddx + ddy*ddy);
    int done = (elapsed >= o.t_limit);
    if(o.max_disp > 0.0 && disp > o.max_disp) done = 1;
    if(!o.reflect && hit) done = 1;
    if(done) break;
  }
  /* FINAL POSITION OF EVERY PARTICLE. The well-mixed gate scores the distribution of
   * these, not touchdowns -- and it is the gate that actually tests the drift, so the
   * port has to expose them. */
  if(fin_x){ fin_x[p] = x; fin_y[p] = y; fin_z[p] = z; }
  atomicAdd((unsigned long long*)&cnt[0], 1ULL);
}

/* ---------------------------------------------------------------- host API ---------- */
extern "C" int lpdmDeviceInit(const LpdmGrid *g, int n_particles_max, long long td_capacity){
  G = *g; NXP = G.nx + 1; NYP = G.ny + 1; NPMAX = n_particles_max; TD_CAP = td_capacity;
  FLDSZ = (size_t)G.nz*NYP*NXP;
  size_t ring = (size_t)LPDM_NFLD*G.nt*FLDSZ;
  CK(cudaMalloc(&fld_d, ring*sizeof(__half)));
  CK(cudaMemset(fld_d, 0, ring*sizeof(__half)));
  CK(cudaMalloc(&ust_d, (size_t)G.nt*NYP*NXP*sizeof(float)));
  CK(cudaMalloc(&z0m_d, (size_t)G.nt*NYP*NXP*sizeof(float)));
  CK(cudaMalloc(&tstamp_d, (size_t)G.nt*sizeof(double)));
  tstamp_h = (double*)calloc(G.nt, sizeof(double));
  CK(cudaMalloc(&zk_d, (size_t)G.nz*sizeof(float)));
  CK(cudaMalloc(&zg_d, (size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMalloc(&sx_d, (size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMalloc(&sy_d, (size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMalloc(&dm_d, (size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMemset(zg_d,0,(size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMemset(sx_d,0,(size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMemset(sy_d,0,(size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMemset(dm_d,0,(size_t)G.ny*G.nx*sizeof(float)));
  CK(cudaMalloc(&flux_d, (size_t)G.ny*G.nx*sizeof(double)));
  CK(cudaMalloc(&conc_d, (size_t)G.ny*G.nx*sizeof(double)));
  CK(cudaMalloc(&sums_d, 2*sizeof(double)));
  CK(cudaMalloc(&cnt_d, 4*sizeof(long long)));
  CK(cudaMalloc(&scratch_d, (size_t)G.nz*G.ny*G.nx*sizeof(float)));
  CK(cudaMalloc(&scratch2_d, (size_t)G.nz*G.ny*G.nx*sizeof(float)));
  if(TD_CAP > 0){
    CK(cudaMalloc(&td_dx_d, TD_CAP*sizeof(float)));
    CK(cudaMalloc(&td_dy_d, TD_CAP*sizeof(float)));
    CK(cudaMalloc(&td_wt_d, TD_CAP*sizeof(float)));
    CK(cudaMalloc(&td_ag_d, TD_CAP*sizeof(float)));
  }
  lpdmDeviceReset();
  return 0;
}

extern "C" void lpdmDeviceReset(void){
  if(flux_d) cudaMemset(flux_d, 0, (size_t)G.ny*G.nx*sizeof(double));
  if(conc_d) cudaMemset(conc_d, 0, (size_t)G.ny*G.nx*sizeof(double));
  if(sums_d) cudaMemset(sums_d, 0, 2*sizeof(double));
  if(cnt_d)  cudaMemset(cnt_d, 0, 4*sizeof(long long));
  if(cov_num_d) cudaMemset(cov_num_d, 0, (size_t)(n_cover>0?n_cover:1)*sizeof(double));
  if(cov_den_d) cudaMemset(cov_den_d, 0, sizeof(double));
}

extern "C" int lpdmDeviceSetStatic(const float *zk, const float *zg, const float *sx,
                                   const float *sy, const float *dmap){
  CK(cudaMemcpy(zk_d, zk, (size_t)G.nz*sizeof(float), cudaMemcpyHostToDevice));
  size_t n2 = (size_t)G.ny*G.nx*sizeof(float);
  have_terrain = 0;
  if(zg){ CK(cudaMemcpy(zg_d, zg, n2, cudaMemcpyHostToDevice));
          for(size_t i=0;i<(size_t)G.ny*G.nx;i++) if(zg[i]!=0.0f){ have_terrain=1; break; } }
  if(sx) CK(cudaMemcpy(sx_d, sx, n2, cudaMemcpyHostToDevice));
  if(sy) CK(cudaMemcpy(sy_d, sy, n2, cudaMemcpyHostToDevice));
  have_dmap = 0;
  if(dmap){ CK(cudaMemcpy(dm_d, dmap, n2, cudaMemcpyHostToDevice));
            for(size_t i=0;i<(size_t)G.ny*G.nx;i++) if(dmap[i]!=0.0f){ have_dmap=1; break; } }
  return 0;
}

extern "C" int lpdmDeviceSetFloor(int n, const double *z, const double *f, int mode){
  floor_mode = mode;
  if(offz_d){ cudaFree(offz_d); cudaFree(offf_d); cudaFree(offs_d);
              offz_d=offf_d=offs_d=NULL; }
  n_off = n; if(n <= 1 || mode == 0){ n_off = 0; return 0; }
  double *sl = (double*)malloc((n-1)*sizeof(double));
  for(int i=0;i<n-1;i++) sl[i] = (f[i+1]-f[i])/(z[i+1]-z[i]);
  CK(cudaMalloc(&offz_d, n*sizeof(double)));
  CK(cudaMalloc(&offf_d, n*sizeof(double)));
  CK(cudaMalloc(&offs_d, (n-1)*sizeof(double)));
  CK(cudaMemcpy(offz_d, z, n*sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(offf_d, f, n*sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(offs_d, sl, (n-1)*sizeof(double), cudaMemcpyHostToDevice));
  free(sl);
  return 0;
}

extern "C" int lpdmDeviceSetCover(int n_mask, const unsigned char *masks){
  if(cover_d){ cudaFree(cover_d); cover_d = NULL; }
  if(cov_num_d){ cudaFree(cov_num_d); cov_num_d = NULL; }
  if(cov_den_d){ cudaFree(cov_den_d); cov_den_d = NULL; }
  n_cover = n_mask; if(n_mask <= 0) return 0;
  size_t n = (size_t)n_mask*G.ny*G.nx;
  CK(cudaMalloc(&cover_d, n));
  CK(cudaMemcpy(cover_d, masks, n, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&cov_num_d, (size_t)n_mask*sizeof(double)));
  CK(cudaMalloc(&cov_den_d, sizeof(double)));
  CK(cudaMemset(cov_num_d, 0, (size_t)n_mask*sizeof(double)));
  CK(cudaMemset(cov_den_d, 0, sizeof(double)));
  return 0;
}

extern "C" int lpdmDevicePushSnapshotHost(int slot, double t_model,
                                          const float *u, const float *v, const float *w,
                                          const float *e, const float *theta,
                                          const float *ustar, const float *z0m){
  size_t n3 = (size_t)G.nz*G.ny*G.nx, n2 = (size_t)G.ny*G.nx;
  int tb = 256;
  int nb3 = (int)((G.nz*(size_t)NYP*NXP + tb - 1)/tb);
  int nb2 = (int)((n2 + tb - 1)/tb);
  const float *src[4] = {u, v, w, e};
  for(int f = 0; f < 4; ++f){
    CK(cudaMemcpy(scratch_d, src[f], n3*sizeof(float), cudaMemcpyHostToDevice));
    kPadTo16<<<nb3, tb>>>(scratch_d, fld_d + (size_t)f*G.nt*FLDSZ + (size_t)slot*FLDSZ,
                          G.nx, G.ny, G.nz, NXP, NYP);
  }
  /* eps and dsig2dz from e and theta, on the device */
  CK(cudaMemcpy(scratch_d, e, n3*sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(scratch2_d, theta, n3*sizeof(float), cudaMemcpyHostToDevice));
  int nbc = (int)((n2 + tb - 1)/tb);
  kDerive<<<nbc, tb>>>(scratch_d, scratch2_d, zk_d,
                       fld_d + (size_t)LPDM_FLD_EPS*G.nt*FLDSZ + (size_t)slot*FLDSZ,
                       fld_d + (size_t)LPDM_FLD_DS2*G.nt*FLDSZ + (size_t)slot*FLDSZ,
                       G.nx, G.ny, G.nz, NXP, NYP, G.dx, G.dy);
  CK(cudaMemcpy(scratch_d, ustar, n2*sizeof(float), cudaMemcpyHostToDevice));
  kPadTo32<<<(int)(((size_t)NYP*NXP + tb - 1)/tb), tb>>>(
      scratch_d, ust_d + (size_t)slot*NYP*NXP, G.nx, G.ny, NXP, NYP);
  CK(cudaMemcpy(scratch_d, z0m, n2*sizeof(float), cudaMemcpyHostToDevice));
  kPadTo32<<<(int)(((size_t)NYP*NXP + tb - 1)/tb), tb>>>(
      scratch_d, z0m_d + (size_t)slot*NYP*NXP, G.nx, G.ny, NXP, NYP);
  (void)nb2;
  tstamp_h[slot] = t_model;
  CK(cudaDeviceSynchronize());
  return 0;
}


/* Push a snapshot whose six fields are ALREADY wrap-padded and ALREADY fp16, exactly as
 * lpdm/fields.py holds them. This is the acceptance path and it exists for one reason:
 * it removes the ingest from the comparison. Test (a) asks whether the GPU INTEGRATOR
 * reproduces the CPU one, so both must be handed bit-identical inputs; pushing raw e and
 * theta and re-deriving eps here would fold a second difference into the same number and
 * a disagreement could not be localised. kDerive is validated separately, on one snapshot,
 * against the Python that wrote these arrays. */
extern "C" int lpdmDevicePushPaddedHalf(int slot, double t_model,
                                        const void *u, const void *v, const void *w,
                                        const void *e, const void *eps, const void *ds2,
                                        const float *ustar_pad, const float *z0m_pad){
  const void *src[LPDM_NFLD] = {u, v, w, e, eps, ds2};
  for(int f = 0; f < LPDM_NFLD; ++f){
    CK(cudaMemcpy(fld_d + (size_t)f*G.nt*FLDSZ + (size_t)slot*FLDSZ, src[f],
                  FLDSZ*sizeof(__half), cudaMemcpyHostToDevice));
  }
  size_t s2 = (size_t)NYP*NXP*sizeof(float);
  CK(cudaMemcpy(ust_d + (size_t)slot*NYP*NXP, ustar_pad, s2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(z0m_d + (size_t)slot*NYP*NXP, z0m_pad, s2, cudaMemcpyHostToDevice));
  tstamp_h[slot] = t_model;
  return 0;
}

/* Derive eps and dsig2dz on the device from interior fp32 e and theta, and hand them
 * straight back as fp32 interior arrays. The ONLY purpose is to score kDerive against
 * lpdm/fields.py on identical input, so the production ingest is validated separately
 * from the integrator instead of the two being confounded. */
extern "C" int lpdmDeviceDeriveCheck(const float *e, const float *theta,
                                     float *eps_out, float *ds2_out){
  size_t n3 = (size_t)G.nz*G.ny*G.nx;
  __half *tmp_eps = NULL, *tmp_ds2 = NULL;
  CK(cudaMalloc(&tmp_eps, FLDSZ*sizeof(__half)));
  CK(cudaMalloc(&tmp_ds2, FLDSZ*sizeof(__half)));
  CK(cudaMemcpy(scratch_d, e, n3*sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(scratch2_d, theta, n3*sizeof(float), cudaMemcpyHostToDevice));
  int tb = 256, nb = (int)(((size_t)G.ny*G.nx + tb - 1)/tb);
  kDerive<<<nb, tb>>>(scratch_d, scratch2_d, zk_d, tmp_eps, tmp_ds2,
                      G.nx, G.ny, G.nz, NXP, NYP, G.dx, G.dy);
  CK(cudaDeviceSynchronize());
  __half *hbuf = (__half*)malloc(FLDSZ*sizeof(__half));
  for(int pass = 0; pass < 2; ++pass){
    CK(cudaMemcpy(hbuf, pass ? tmp_ds2 : tmp_eps, FLDSZ*sizeof(__half),
                  cudaMemcpyDeviceToHost));
    float *dst = pass ? ds2_out : eps_out;
    for(int k = 0; k < G.nz; ++k) for(int j = 0; j < G.ny; ++j) for(int i = 0; i < G.nx; ++i)
      dst[((size_t)k*G.ny + j)*G.nx + i] =
        __half2float(hbuf[(size_t)k*NYP*NXP + (size_t)j*NXP + i]);
  }
  free(hbuf); cudaFree(tmp_eps); cudaFree(tmp_ds2);
  return 0;
}

extern "C" int lpdmDeviceRelease(const LpdmOpts *o, int n,
                                 const double *xr_a, const double *yr_a,
                                 const double *zr_a, const double *t_rel,
                                 double x_ref, double y_ref,
                                 double *fin_x, double *fin_y, double *fin_z){
  double *t_d=NULL,*x_d=NULL,*y_d=NULL,*z_d=NULL,*fx_d=NULL,*fy_d=NULL,*fz_d=NULL;
  size_t nb = (size_t)n*sizeof(double);
  CK(cudaMalloc(&t_d, nb)); CK(cudaMalloc(&x_d, nb));
  CK(cudaMalloc(&y_d, nb)); CK(cudaMalloc(&z_d, nb));
  CK(cudaMemcpy(t_d, t_rel, nb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(x_d, xr_a, nb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(y_d, yr_a, nb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(z_d, zr_a, nb, cudaMemcpyHostToDevice));
  if(fin_z){ CK(cudaMalloc(&fx_d, nb)); CK(cudaMalloc(&fy_d, nb));
             CK(cudaMalloc(&fz_d, nb)); }
  Ctx c;
  c.nx=G.nx; c.ny=G.ny; c.nz=G.nz; c.nt=G.nt; c.nxp=NXP; c.nyp=NYP;
  c.dx=G.dx; c.dy=G.dy; c.x0=G.x0; c.y0=G.y0; c.zC=G.zC; c.dt_dump=G.dt_dump;
  c.t0_ring = tstamp_h[0];
  c.fld=fld_d; c.ust=ust_d; c.z0m=z0m_d;
  c.zk=zk_d; c.zg=zg_d; c.sx=sx_d; c.sy=sy_d; c.dm=dm_d;
  c.have_terrain=have_terrain; c.have_dmap=have_dmap;
  c.offz=offz_d; c.offf=offf_d; c.offs=offs_d; c.n_off=n_off; c.floor_mode=floor_mode;
  float zk0, zkn;
  CK(cudaMemcpy(&zk0, zk_d, sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&zkn, zk_d+(G.nz-1), sizeof(float), cudaMemcpyDeviceToHost));
  c.z_ref = zk0; c.z_top = zkn;
  double i_r = ((x_ref - G.x0)/G.dx), j_r = ((y_ref - G.y0)/G.dy);
  int tb = 128, nblk = (n + tb - 1)/tb;
  kIntegrate<<<nblk, tb>>>(c, *o, n, x_d, y_d, z_d, t_d, x_ref, y_ref, i_r, j_r,
                         fx_d, fy_d, fz_d,
                         flux_d, conc_d, sums_d, cnt_d, cover_d, n_cover,
                         cov_num_d, cov_den_d, td_dx_d, td_dy_d, td_wt_d, td_ag_d, TD_CAP);
  CK(cudaDeviceSynchronize());
  cudaError_t er = cudaGetLastError();
  if(er != cudaSuccess){ fprintf(stderr,"LPDM kernel: %s\n", cudaGetErrorString(er));
                         cudaFree(t_d); cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
                         return -1; }
  if(fin_z){
    CK(cudaMemcpy(fin_x, fx_d, nb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(fin_y, fy_d, nb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(fin_z, fz_d, nb, cudaMemcpyDeviceToHost));
    cudaFree(fx_d); cudaFree(fy_d); cudaFree(fz_d);
  }
  cudaFree(t_d); cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
  return 0;
}

extern "C" int lpdmDeviceFetch(double *flux, double *conc, double *sf, double *sc,
                               long long *np_, long long *ntd,
                               double *cov_num, double *cov_den){
  size_t n2 = (size_t)G.ny*G.nx;
  CK(cudaMemcpy(flux, flux_d, n2*sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(conc, conc_d, n2*sizeof(double), cudaMemcpyDeviceToHost));
  double s[2]; long long ct[4];
  CK(cudaMemcpy(s, sums_d, 2*sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ct, cnt_d, 4*sizeof(long long), cudaMemcpyDeviceToHost));
  *sf = s[0]; *sc = s[1]; *np_ = ct[0]; *ntd = ct[1];
  if(n_cover > 0 && cov_num){
    CK(cudaMemcpy(cov_num, cov_num_d, (size_t)n_cover*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cov_den, cov_den_d, sizeof(double), cudaMemcpyDeviceToHost));
  }
  return 0;
}

extern "C" long long lpdmDeviceFetchTouchdowns(long long cap, float *dx, float *dy,
                                               float *wt, float *age, long long *n_total,
                                               int *overflowed){
  long long ct[4];
  cudaMemcpy(ct, cnt_d, 4*sizeof(long long), cudaMemcpyDeviceToHost);
  long long kept = ct[2] < TD_CAP ? ct[2] : TD_CAP;
  if(kept > cap) kept = cap;
  if(kept > 0 && dx){
    cudaMemcpy(dx, td_dx_d, kept*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dy, td_dy_d, kept*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wt, td_wt_d, kept*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(age, td_ag_d, kept*sizeof(float), cudaMemcpyDeviceToHost);
  }
  if(n_total) *n_total = ct[2];
  if(overflowed) *overflowed = (int)ct[3];
  return kept;
}

extern "C" void lpdmDeviceFree(void){
  if(fld_d) cudaFree(fld_d); if(ust_d) cudaFree(ust_d); if(z0m_d) cudaFree(z0m_d);
  if(tstamp_d) cudaFree(tstamp_d); if(tstamp_h) free(tstamp_h);
  if(zk_d) cudaFree(zk_d); if(zg_d) cudaFree(zg_d); if(sx_d) cudaFree(sx_d);
  if(sy_d) cudaFree(sy_d); if(dm_d) cudaFree(dm_d);
  if(offz_d) cudaFree(offz_d); if(offf_d) cudaFree(offf_d); if(offs_d) cudaFree(offs_d);
  if(cover_d) cudaFree(cover_d); if(cov_num_d) cudaFree(cov_num_d);
  if(cov_den_d) cudaFree(cov_den_d);
  if(flux_d) cudaFree(flux_d); if(conc_d) cudaFree(conc_d);
  if(sums_d) cudaFree(sums_d); if(cnt_d) cudaFree(cnt_d);
  if(scratch_d) cudaFree(scratch_d); if(scratch2_d) cudaFree(scratch2_d);
  if(td_dx_d) cudaFree(td_dx_d); if(td_dy_d) cudaFree(td_dy_d);
  if(td_wt_d) cudaFree(td_wt_d); if(td_ag_d) cudaFree(td_ag_d);
  fld_d=NULL; ust_d=NULL; z0m_d=NULL; tstamp_d=NULL; tstamp_h=NULL;
  zk_d=zg_d=sx_d=sy_d=dm_d=NULL; offz_d=offf_d=offs_d=NULL;
  cover_d=NULL; cov_num_d=cov_den_d=NULL; flux_d=conc_d=NULL; sums_d=NULL; cnt_d=NULL;
  scratch_d=scratch2_d=NULL; td_dx_d=td_dy_d=td_wt_d=td_ag_d=NULL;
}
