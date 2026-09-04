# FastEddy v5.0.1 toolchain image — Kegonsa Solar Array flux footprint project
#
# Contains ONLY the toolchain. FastEddy source is NOT copied in; it is compiled
# inside the bind-mounted host tree (see docker/build_fasteddy.sh) so the patched tree's
# git working tree stays authoritative and source edits never rebuild the image.
#
# Dependency set follows inst.txt's FastEddy block. Two deviations from it, both
# deliberate:
#
#   * inst.txt starts with `apt install nvidia-cuda-toolkit`, which on Ubuntu
#     22.04 is CUDA 11.5. CUDA 11.5 predates sm_89 entirely -- the RTX 4080
#     cannot be targeted with it. The 11.8 base image supplies the toolkit
#     instead. 11.8 is the FIRST release with sm_89 support, so the PROJECT_BRIEF.md
#     pin is a hard floor, not a preference.
#
#   * inst.txt installs gcc-13/gfortran-13 from the toolchain PPA. nvcc 11.8
#     supports gcc <= 11, so Ubuntu 22.04's default gcc-11 is used. (The gcc-13
#     line in inst.txt belongs to its MPAS/LLVM-Flang sections, not FastEddy.)
#
# inst.txt's ParallelIO, PnetCDF and netcdf-fortran entries are likewise MPAS
# work: FastEddy links only -lm -lmpi -lstdc++ -lcurand -lcudart -lnetcdf.
# They are installed anyway below, exactly as inst.txt lists them, because they
# are cheap and keep this image faithful to the notes that are known to work.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Layer 1: compilers, MPI, NetCDF/HDF5 — inst.txt's apt block.
#
# Stock Ubuntu OpenMPI (4.1.2) is used deliberately. It reports
# mpi_built_with_cuda_support:false, and FastEddy DOES hand raw device pointers
# to MPI_Isend/MPI_Irecv in SRC/FECUDA/fecuda_Utils.cu under hydroBCs==2
# (periodic — our configuration), with no numProcs==1 bypass. That looks like it
# should require a CUDA-aware MPI, so it was tested directly rather than assumed:
#
#   A CUDA-aware OpenMPI 4.1.6 was built from source with --with-cuda and the
#   Example03_SBL smoke case was run against both builds. Field differences at
#   t=200 steps were statistically indistinguishable from the model's own
#   run-to-run nondeterminism (rho 3.70e-6 vs 3.81e-6 baseline; theta 7.0e-4 vs
#   6.4e-4 baseline). At one rank the MPI build makes no difference.
#
# So: stock OpenMPI, which also builds in ~2 minutes instead of ~20. If this
# project ever goes multi-GPU, revisit — that is where CUDA-awareness would
# actually be exercised.
#
# Fortran bindings are REQUIRED even though FastEddy has no Fortran source at
# all: its C code broadcasts with MPI_INTEGER and MPI_CHARACTER (60 occurrences
# across 8 files, e.g. SRC/IO/io.c:128-229) rather than the C MPI_INT/MPI_CHAR.
# Those are Fortran datatype handles. Building MPI with --disable-mpi-fortran
# removes them and the model aborts at startup with
# "MPI_ERR_TYPE: invalid datatype" in MPI_Bcast. Verified by doing exactly that.
# Stock libopenmpi-dev ships Fortran bindings, so inst.txt's recipe sidesteps
# this automatically.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        make \
        wget \
        ca-certificates \
        file \
        openmpi-bin \
        libopenmpi-dev \
        libhdf5-openmpi-dev \
        libnetcdf-dev \
        libnetcdff-dev \
        libpnetcdf-dev \
        pnetcdf-bin \
        netcdf-bin \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Fail the build here rather than at model startup.
RUN ompi_info | grep -q "Fort mpif.h: yes" \
    || { echo "FATAL: MPI lacks Fortran bindings; FastEddy's MPI_INTEGER/MPI_CHARACTER broadcasts would fail."; exit 1; }

# ---------------------------------------------------------------------------
# Layer 2: Python for FastEddy's pre/post-processing utilities.
#
# Mirrors scripts/batch_jobs/environment.yml, PLUS scikit-image: SimGrid.py does
# `from skimage.measure import block_reduce` but NCAR's own environment.yml
# omits it. mpi4py compiles against the OpenMPI installed above.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        numpy \
        scipy \
        matplotlib \
        netCDF4 \
        xarray \
        pandas \
        dask \
        scikit-image \
        mpi4py

# ---------------------------------------------------------------------------
# Layer 3: HRRR pseudo-sounding retrieval (bin/hrrr_sounding.py).
#
# The per-case forcing comes from HRRR analyses rather than CONUS404, which
# carries no atmospheric profiles at all (PROJECT_BRIEF.md). Herbie does the archive
# lookup and GRIB byte-range subsetting; cfgrib/eccodes decode the messages.
#
# `eccodes` is the pip binary wheel, which ships its own libeccodes -- the apt
# package is NOT installed, because the two disagree on definitions paths and
# cfgrib then picks up whichever it finds first. One source of eccodes only.
#
# Pinned to a floor, not an exact version: Herbie's archive source list changes
# as NOAA moves buckets, and an old pin silently loses access to date ranges.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        "eccodes>=1.7" \
        "cfgrib>=0.9.10" \
        "herbie-data>=2024.3.0" \
        "pyproj>=3.6" \
        s3fs \
        scikit-learn \
    && python3 -c "import cfgrib, herbie; print('cfgrib', cfgrib.__version__, 'herbie', herbie.__version__)"

# Consumed by docker/build_fasteddy.sh to supply the include/library paths that
# FastEddy's Makefile expects NCAR's module wrapper to inject.
#
# OMPI_MCA_plm=isolated: OpenMPI's default launcher is `rsh`, which aborts
# hunting for an ssh binary a single-node container has no use for. `isolated`
# is the correct single-node launcher. Override with --mca plm rsh if needed.
ENV CUDA_HOME=/usr/local/cuda \
    MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi \
    NETCDF_ROOT=/usr \
    OMPI_MCA_plm=isolated

WORKDIR /work
