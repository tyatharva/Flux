/* FastEddy(R) fork: GPU-resident backward LPDM -- device interface.
 *
 * WHY THIS EXISTS. The offline path writes every LPDM field to disk at a 5 s cadence and
 * the analysis reads them back: 18.4 MB per dump, ~16.6 GB per corpus case, ~2000 cases.
 * On this machine that is merely slow; on rented GPUs the scratch tier is the single
 * largest unknown in the corpus plan. This module keeps t_back of history in a VRAM ring
 * buffer and integrates the backward ensemble against it in-kernel, so a case's entire
 * output is the footprint plus a touchdown sample -- about 1.6 MB.
 *
 * IT IS A TRANSLITERATION OF lpdm/model.py AND lpdm/fields.py, NOT A REIMPLEMENTATION.
 * Every constant, floor, clip and sub-layer continuation below has a line in the Python
 * it was copied from, and bin/test_gpu_lpdm.py scores this against that Python on the
 * SAME saved fields. A CUDA rewrite of a Langevin integrator with drift terms is exactly
 * where well-mixedness breaks, and it breaks silently -- so the port ships only if it
 * matches, and the acceptance suite is the whole point rather than a formality.
 *
 * PRECISION. Fields are fp16 in the ring (the offline cache already is, and ioLPDMmode's
 * 16-bit packing measured harmless: 0 m in peak, 19 m in centroid against a 59.2% error
 * floor). PARTICLE STATE IS fp64 -- positions integrate for thousands of steps and fp32
 * roundoff accumulates into a spurious drift (PROJECT_BRIEF.md convention).
 */
#ifndef _CUDA_LPDMDEVICE_CU_H
#define _CUDA_LPDMDEVICE_CU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Ring-buffer field order. eps and dsig2dz are DERIVED on ingest, exactly as
 * lpdm/fields.py derives them, rather than stored by the LES: they depend on theta
 * through the Ozmidov-limited mixing length, so deriving them once per snapshot costs one
 * cheap kernel and keeps the closure identical to the offline path. */
#define LPDM_FLD_U   0
#define LPDM_FLD_V   1
#define LPDM_FLD_W   2
#define LPDM_FLD_E   3
#define LPDM_FLD_EPS 4
#define LPDM_FLD_DS2 5
#define LPDM_NFLD    6

typedef struct {
  int   nx, ny, nz;      /* INTERIOR extents (no halos), matching the netCDF output      */
  int   nt;              /* ring depth in snapshots; must cover t_back                   */
  float dx, dy;          /* horizontal spacing, m                                        */
  float x0, y0;          /* cell-centre position of (i,j) = (0,0), m                     */
  float zC;              /* terrain-following ceiling from the zDeform inversion, m      */
  float dt_dump;         /* seconds between ring snapshots                               */
} LpdmGrid;

typedef struct {
  double c0;             /* Weil/Sullivan/Moeng Langevin constant (3.0)                  */
  double z_touch;        /* touchdown height above ground, m (2.0)                       */
  double dt_frac;        /* adaptive step as a fraction of T_L (0.05)                    */
  double dt_min, dt_max; /* adaptive step bounds, s (0.01, 1.0)                          */
  double t_limit;        /* t_back, s                                                    */
  double max_disp;       /* retire beyond this unwrapped displacement, m; <=0 disables   */
  double w_floor;        /* bound on the 2/|w_td| weight, m/s (0.02)                     */
  int    reflect;        /* 1 = reflect at touchdown (production)                        */
  int    direction;      /* -1 backward, +1 forward                                      */
  int    max_iter;
  double td_prob;        /* Bernoulli keep-probability for the touchdown sample          */
  /* DOUBLE-ROTATION ANGLES (Wilczak et al. 2001). The flux weight is the STREAMLINE-frame
   * vertical velocity at release, not the model-frame one, because that is the frame an
   * eddy-covariance instrument reports in. Computed on the host from the window's own mean
   * wind, exactly as lpdm/driver.py does: theta = atan2(Vb, Ub), phi = atan2(Wb, |U_h|). */
  double yaw, pitch;
  unsigned long long seed;
} LpdmOpts;

/* One-time device allocation. Returns 0 on success. */
int lpdmDeviceInit(const LpdmGrid *g, int n_particles_max, long long td_capacity);

/* Static per-column geometry. All arrays are (ny,nx) row-major except zk, which is
 * (nz) flat-column heights F_k from the zDeform inversion. dmap may be NULL. */
int lpdmDeviceSetStatic(const float *zk, const float *zg, const float *slope_x,
                        const float *slope_y, const float *dmap);

/* The sigma^2 floor as a 1-D ADDITIVE column offset, delta(z) -- the production delivery
 * (lpdm/sgs_floor.py). Computed on the host, because it is a function of window-mean
 * statistics rather than of the instantaneous field, and handing the same table to both
 * paths is what makes the acceptance comparison a comparison of the INTEGRATOR. */
int lpdmDeviceSetFloor(int n, const double *z, const double *f, int mode);
/* mode: 0 = no floor, 1 = MULTIPLICATIVE sc(z) (production), 2 = additive delta(z). */

/* Land-cover masks for the footprint-weighted share, (ny,nx) 0/1, in the LES index space
 * the offline path attributes in. `names` is only carried back out for labelling. */
int lpdmDeviceSetCover(int n_mask, const unsigned char *masks);

/* Push one snapshot into ring slot (step index modulo nt). Host pointers are interior
 * (nz,ny,nx) fp32; the wrap-pad, the fp16 conversion and the eps/dsig2dz derivation all
 * happen on the device. `t_model` is the snapshot's model time in seconds. */
int lpdmDevicePushSnapshotHost(int slot, double t_model,
                               const float *u, const float *v, const float *w,
                               const float *e, const float *theta,
                               const float *ustar, const float *z0m);

/* Release one ensemble and integrate it to completion. Touchdowns are deposited on the
 * device (cloud-in-cell, signed) and a bounded uniform subsample is retained. */
/* Per-particle release positions, so the same kernel serves both the footprint (all
 * particles at the receptor) and the well-mixed gate (a uniformly filled column).
 * (x_ref, y_ref) is the origin the raster and the touchdown sample are relative to.
 * fin_* may be NULL; when given they receive every particle's final position. */
int lpdmDeviceRelease(const LpdmOpts *o, int n_particles,
                      const double *x_rel, const double *y_rel, const double *z_rel,
                      const double *t_rel, double x_ref, double y_ref,
                      double *fin_x, double *fin_y, double *fin_z);

/* Results. `flux`/`conc` are (ny,nx) accumulators in the SAME index space as the LES. */
int lpdmDeviceFetch(double *flux, double *conc, double *sum_flux_all, double *sum_conc_all,
                    long long *n_particles, long long *n_touchdown,
                    double *cover_num, double *cover_den);
/* Touchdown sample: unfolded receptor-relative dx, dy; signed weight; age. */
long long lpdmDeviceFetchTouchdowns(long long cap, float *dx, float *dy, float *wt,
                                    float *age, long long *n_total, int *overflowed);
void lpdmDeviceReset(void);
void lpdmDeviceFree(void);

#ifdef __cplusplus
}
#endif
#endif
