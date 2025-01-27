#!/usr/bin/env bash

set -e
set -x

. /opt/spack/share/spack/setup-env.sh
spack env activate trilinos

cd /opt/build/Trilinos

export MPI_ROOT="$(dirname $(which mpicc))"
export MPICC="${MPI_ROOT}/mpicc"
export MPICXX="${MPI_ROOT}/mpicxx"
export MPIF90="${MPI_ROOT}/mpif90"
export MPIRUN="${MPI_ROOT}/mpirun"

export BLAS_ROOT="$(spack location -i openblas)"
export LAPACK_ROOT="${BLAS_ROOT}"

export CUDA_ROOT=/usr/local/cuda
export PATH=${CUDA_ROOT}/bin:$PATH
export OMPI_CXX=/opt/src/Trilinos/packages/kokkos/bin/nvcc_wrapper
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
ENABLE_CUDA=ON

cmake -G "${CMAKE_GENERATOR:-Ninja}" \
    -D CMAKE_BUILD_TYPE=DEBUG \
    -D Trilinos_ENABLE_DEBUG=ON \
    -D Trilinos_PARALLEL_LINK_JOBS_LIMIT=2 \
    -D Trilinos_ENABLE_ALL_PACKAGES=ON \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=ON \
    -D Trilinos_ALLOW_NO_PACKAGES=ON \
    -D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
    -D Trilinos_IGNORE_MISSING_EXTRA_REPOSITORIES=ON \
    -D Trilinos_ENABLE_TESTS=ON \
    -D Trilinos_TEST_CATEGORIES=BASIC \
    -D Trilinos_ENABLE_ALL_FORWARD_DEP_PACKAGES=ON \
    -D Trilinos_VERBOSE_CONFIGURE=ON \
    -D BUILD_SHARED_LIBS=ON \
    \
    -D Trilinos_WARNINGS_AS_ERRORS_FLAGS="-Wno-error" \
    -D Trilinos_ENABLE_SEACAS=OFF \
    -D Trilinos_ENABLE_Sacado=OFF \
    \
    -D TPL_ENABLE_CUDA="${ENABLE_CUDA}" \
    -D Tpetra_INST_SERIAL=ON \
    -D Tpetra_INST_CUDA=ON \
    -D Trilinos_ENABLE_Kokkos=ON \
    -D Kokkos_ARCH_AMPERE86=ON \
    -D Kokkos_ENABLE_OPENMP=OFF \
    -D Kokkos_ENABLE_CUDA="${ENABLE_CUDA}" \
    -D Kokkos_ENABLE_CUDA_LAMBDA="${ENABLE_CUDA}" \
    -D Kokkos_ENABLE_CUDA_UVM=OFF \
    \
    -D TPL_ENABLE_CUSOLVER=ON \
    -D TPL_ENABLE_CUSPARSE=ON \
    \
    -D TPL_ENABLE_BLAS=ON \
    -D TPL_BLAS_LIBRARIES="${BLAS_ROOT}/lib/libopenblas.so" \
    -D TPL_ENABLE_LAPACK=ON \
    -D TPL_LAPACK_LIBRARIES="${LAPACK_ROOT}/lib/libopenblas.so" \
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
    -D CMAKE_C_COMPILER=${MPICC} \
    -D CMAKE_CXX_COMPILER=${MPICXX} \
    -D CMAKE_Fortran_COMPILER=${MPIF90} \
    -D TPL_ENABLE_MPI=ON \
    -D MPI_BIN_DIR=${MPIRUN} \
    -D MPI_EXEC=${MPIRUN} \
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
ninja -j 4
