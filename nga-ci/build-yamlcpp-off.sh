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

export NETCDF_ROOT="$(spack location -i netcdf-c)"
export MATIO_ROOT="$(spack location -i matio)"
export HDF5_ROOT="$(spack location -i hdf5)"
export BOOST_ROOT="$(spack location -i boost)"

cmake -G "${CMAKE_GENERATOR:-Ninja}" \
    -D CMAKE_BUILD_TYPE=DEBUG \
    -D Trilinos_ENABLE_DEBUG=ON \
    -D Trilinos_PARALLEL_LINK_JOBS_LIMIT=3 \
    -D Trilinos_ENABLE_ALL_PACKAGES=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=ON \
    -D Trilinos_ALLOW_NO_PACKAGES=ON \
    -D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
    -D Trilinos_IGNORE_MISSING_EXTRA_REPOSITORIES=ON \
    -D Trilinos_ENABLE_TESTS=ON \
    -D Trilinos_TEST_CATEGORIES=BASIC \
    -D Trilinos_ENABLE_ALL_FORWARD_DEP_PACKAGES=ON \
    -D Trilinos_VERBOSE_CONFIGURE=OFF \
    -D BUILD_SHARED_LIBS:BOOL=ON \
    -D Tpetra_ENABLE_DEPRECATED_CODE=ON \
    \
    -D Trilinos_ENABLE_Panzer=ON \
    -D Trilinos_ENABLE_PanzerMiniEM=ON \
    \
    \
    -D TPL_ENABLE_BLAS=ON \
    -D TPL_BLAS_LIBRARIES="${BLAS_ROOT}/lib/libopenblas.so" \
    -D TPL_ENABLE_LAPACK=ON \
    -D TPL_LAPACK_LIBRARIES="${LAPACK_ROOT}/lib/libopenblas.so" \
    \
    \
    -D TPL_ENABLE_Boost=ON \
    -D Boost_LIBRARY_DIRS="${BOOST_ROOT}/lib" \
    -D Boost_INCLUDE_DIRS="${BOOST_ROOT}/include" \
    \
    -D TPL_ENABLE_Matio=ON \
    -D Matio_LIBRARY_DIRS="${MATIO_ROOT}/lib" \
    -D Matio_INCLUDE_DIRS="${MATIO_ROOT}/include" \
    \
    -D TPL_ENABLE_Netcdf=ON \
    -D Netcdf_LIBRARY_NAMES=netcdf \
    -D Netcdf_LIBRARY_DIRS="${NETCDF_ROOT}/lib" \
    -D TPL_Netcdf_INCLUDE_DIRS="${NETCDF_ROOT}/include" \
    \
    -D TPL_Netcdf_PARALLEL=OFF \
    \
    -D TPL_ENABLE_HDF5=ON \
    -D HDF5_LIBRARY_DIRS="${HDF5_ROOT}/lib" \
    -D HDF5_INCLUDE_DIRS="${HDF5_ROOT}/include" \
    -D TPL_ENABLE_Matio=OFF \
    \
    -D TPL_ENABLE_CUDA=OFF \
    \
    -D Trilinos_MUST_FIND_ALL_TPL_LIBS=OFF \
    -D TPL_ENABLE_yaml-cpp=OFF \
    -D TPL_ENABLE_X11=OFF \
    -D TPL_ENABLE_Pthread=OFF \
    -D TPL_ENABLE_BoostLib=OFF \
    -D TPL_ENABLE_ParMETIS=OFF \
    -D TPL_ENABLE_Zlib=OFF \
    -D TPL_ENABLE_SuperLU=OFF \
    -D TPL_ENABLE_Scotch=OFF \
    \
    \
    -D CMAKE_C_COMPILER=${MPICC} \
    -D CMAKE_CXX_COMPILER=${MPICXX} \
    -D CMAKE_Fortran_COMPILER=${MPIF90} \
    -D TPL_ENABLE_MPI=ON \
    -D MPI_EXEC=${MPIRUN} \
    -D MPI_EXEC_MAX_NUMPROCS=4 \
    \
    \
    -S /opt/src/Trilinos -B /opt/build/Trilinos
ninja -j 12
