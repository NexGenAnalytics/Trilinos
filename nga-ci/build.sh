#!/usr/bin/env bash

set -e
set -x
. /opt/spack/share/spack/setup-env.sh
spack env activate trilinos

spack find -p

export MPICC_DIR=$(which mpicc)
export MPICXX_DIR=$(which mpicxx)
export MPIF90_DIR=$(which mpif90)
export MPIRUN_DIR=$(which mpirun)

export BLAS_ROOT="/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-11.4.0/openblas-0.3.23-bwv7xuj5t72zlgxhiq4wz3nyb35b2two"
export LAPACK_ROOT="/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-11.4.0/openblas-0.3.23-bwv7xuj5t72zlgxhiq4wz3nyb35b2two"

cmake -G "${CMAKE_GENERATOR:-Ninja}" \
    -D CMAKE_PREFIX_PATH=/opt \
    -D Trilinos_PARALLEL_LINK_JOBS_LIMIT=2 \
    -D Trilinos_ENABLE_ALL_PACKAGES=ON \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=ON \
    -D Trilinos_ALLOW_NO_PACKAGES=ON \
    -D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
    -D Trilinos_IGNORE_MISSING_EXTRA_REPOSITORIES=ON \
    -D Trilinos_ENABLE_TESTS=ON \
    -D Trilinos_TEST_CATEGORIES=BASIC \
    -D Trilinos_ENABLE_ALL_FORWARD_DEP_PACKAGES=ON \
    \
    -D TPL_ENABLE_BLAS=ON \
    -D BLAS_LIBRARY_DIRS="${BLAS_ROOT}/lib" \
    -D TPL_ENABLE_LAPACK=ON \
    -D LAPACK_LIBRARY_DIRS="${LAPACK_ROOT}/lib" \
    \
    -D TPL_ENABLE_Matio=OFF \
    -D TPL_ENABLE_X11=OFF \
    -D TPL_ENABLE_Pthread=OFF \
    -D TPL_ENABLE_Boost=OFF \
    -D TPL_ENABLE_BoostLib=OFF \
    -D TPL_ENABLE_ParMETIS=OFF \
    -D TPL_ENABLE_Zlib=OFF \
    -D TPL_ENABLE_HDF5=OFF \
    -D TPL_ENABLE_Netcdf=OFF \
    -D TPL_ENABLE_SuperLU=OFF \
    -D TPL_ENABLE_Scotch=OFF \
    \
    -D CMAKE_C_COMPILER=${MPICC_DIR} \
    -D CMAKE_CXX_COMPILER=${MPICXX_DIR} \
    -D CMAKE_Fortran_COMPILER=${MPIF90_DIR} \
    -D TPL_ENABLE_MPI=ON \
    -D MPI_EXEC=${MPIRUN_DIR} \
    \
    -D Trilinos_ENABLE_Rythmos=OFF \
    -D Trilinos_ENABLE_Pike=OFF \
    -D Trilinos_ENABLE_Komplex=OFF \
    -D Trilinos_ENABLE_TriKota=OFF \
    -D Trilinos_ENABLE_Moertel=OFF \
    -D Trilinos_ENABLE_Domi=OFF \
    -D Trilinos_ENABLE_FEI=OFF \
    \
    -D Trilinos_ENABLE_PyTrilinos=OFF \
    \
    -D Trilinos_ENABLE_Epetra=OFF \
    -S /opt/src/Trilinos -B /opt/build/Trilinos
ninja -j6