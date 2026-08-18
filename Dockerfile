# FastEddy v5.0.1 toolchain image — Kegonsa Solar Array flux footprint project
#
# Contains ONLY the toolchain. FastEddy source is NOT copied in; it is compiled
# inside the bind-mounted host tree (see docker/build_fasteddy.sh) so that the
# fork's git working tree stays authoritative and source edits never trigger an
# image rebuild.
#
# CUDA 11.8 is a floor, not a preference: it is the first toolkit release with
# sm_89 (Ada / RTX 4080) support.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Layer 1: system toolchain + NetCDF-C.
#
# Ubuntu 22.04's default gcc-11 is REQUIRED, not incidental: nvcc 11.8 supports
# gcc <= 11. (inst.txt's gcc-13 would have been rejected by nvcc.)
#
# FastEddy links exactly: -lm -lmpi -lstdc++ -lcurand -lcudart -lnetcdf
# so there is deliberately no Fortran compiler, no PnetCDF, and no ParallelIO
# here despite inst.txt installing all three -- none of them are used.
#
# libnetcdf-dev must be NetCDF-4 capable because SRC/IO/io_netcdf.c forces
# NC_NETCDF4 on nc_create(); that is why libhdf5-dev comes along.
# netcdf-bin supplies ncdump, used to verify model output.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        wget \
        ca-certificates \
        file \
        libnetcdf-dev \
        netcdf-bin \
        libhdf5-dev \
        libnuma-dev \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Layer 2: CUDA-aware OpenMPI, built from source.
#
# This is NOT optional and NOT only for multi-GPU runs. In
# SRC/FECUDA/fecuda_Utils.cu, under hydroBCs==2 (periodic -- every tutorial and
# our own configuration), every rank exchanges halos unconditionally, including
# a single rank sending to itself, passing raw *device* pointers to
# MPI_Isend/MPI_Irecv. There is no numProcs==1 bypass. Stock Ubuntu OpenMPI is
# built without CUDA support and would dereference those device pointers as
# host memory.
#
# --disable-mpi-fortran: FastEddy is C-only, and this cuts build time.
# UCX is not built: it is only needed for multi-node transports, and this
# project runs on a single GPU.
# ---------------------------------------------------------------------------
ARG OMPI_VERSION=4.1.6
ARG OMPI_SERIES=v4.1
RUN wget -q "https://download.open-mpi.org/release/open-mpi/${OMPI_SERIES}/openmpi-${OMPI_VERSION}.tar.bz2" \
    && tar xf "openmpi-${OMPI_VERSION}.tar.bz2" \
    && cd "openmpi-${OMPI_VERSION}" \
    && ./configure \
        --prefix=/opt/openmpi \
        --with-cuda=/usr/local/cuda \
        --disable-mpi-fortran \
        --enable-mpi1-compatibility \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf "openmpi-${OMPI_VERSION}" "openmpi-${OMPI_VERSION}.tar.bz2"

ENV PATH=/opt/openmpi/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/openmpi/lib:${LD_LIBRARY_PATH}

# Fail the image build here rather than discovering it as a segfault mid-run.
RUN ompi_info --parsable --all | grep -q "mpi_built_with_cuda_support:value:true" \
    || { echo "FATAL: OpenMPI was built WITHOUT CUDA support; FastEddy halo exchange would segfault."; exit 1; }

# ---------------------------------------------------------------------------
# Layer 3: Python for FastEddy's pre/post-processing utilities.
#
# Mirrors scripts/batch_jobs/environment.yml, PLUS scikit-image: SimGrid.py
# does `from skimage.measure import block_reduce` but NCAR's own environment.yml
# omits it.
# mpi4py is compiled here against the CUDA-aware OpenMPI installed above.
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

# Consumed by docker/build_fasteddy.sh to fill in the include/lib paths that
# FastEddy's Makefile expects NCAR's module wrapper to supply.
ENV CUDA_HOME=/usr/local/cuda \
    MPI_ROOT=/opt/openmpi \
    NETCDF_ROOT=/usr

WORKDIR /work
