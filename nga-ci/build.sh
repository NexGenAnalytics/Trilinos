#!/usr/bin/env bash

set -e
set -x
. /opt/spack/share/spack/setup-env.sh
spack env activate trilinos

export MPICC_DIR=$(which mpicc)
export MPICXX_DIR=$(which mpicxx)
export MPIF90_DIR=$(which mpif90)
export MPIRUN_DIR=$(which mpirun)

cmake -G "Ninja" \
    -D CMAKE_GENERATOR STRING : Ninja \
    -D Trilinos_PARALLEL_LINK_JOBS_LIMIT STRING : 2 \
    -D Trilinos_ENABLE_ALL_PACKAGES BOOL : ON \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES BOOL : ON \
    -D Trilinos_ALLOW_NO_PACKAGES BOOL : ON \
    -D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES BOOL : ON \
    -D Trilinos_IGNORE_MISSING_EXTRA_REPOSITORIES BOOL : ON \
    -D Trilinos_ENABLE_TESTS BOOL : ON \
    -D Trilinos_TEST_CATEGORIES STRING : BASIC \
    -D Trilinos_ENABLE_ALL_FORWARD_DEP_PACKAGES BOOL : ON \
    \
    -D TPL_ENABLE_Matio    BOOL : OFF \
    -D TPL_ENABLE_X11      BOOL : OFF \
    -D TPL_ENABLE_Pthread  BOOL : OFF \
    -D TPL_ENABLE_BLAS     BOOL : ON \
    -D TPL_ENABLE_LAPACK   BOOL : ON \
    -D TPL_ENABLE_Boost    BOOL : OFF \
    -D TPL_ENABLE_BoostLib BOOL : OFF \
    -D TPL_ENABLE_ParMETIS BOOL : OFF \
    -D TPL_ENABLE_Zlib     BOOL : OFF \
    -D TPL_ENABLE_HDF5     BOOL : OFF \
    -D TPL_ENABLE_Netcdf   BOOL : OFF \
    -D TPL_ENABLE_SuperLU  BOOL : OFF \
    -D TPL_ENABLE_Scotch   BOOL : OFF \
    \
    -D CMAKE_C_COMPILER       FILEPATH : ${MPICC_DIR} \
    -D CMAKE_CXX_COMPILER     FILEPATH : ${MPICXX_DIR} \
    -D CMAKE_Fortran_COMPILER FILEPATH : ${MPIF90_DIR} \
    -D TPL_ENABLE_MPI         BOOL     : ON \
    -D MPI_EXEC               FILEPATH : ${MPIRUN_DIR} \
    \
    -D Trilinos_ENABLE_Rythmos BOOL : OFF \
    -D Trilinos_ENABLE_Pike    BOOL : OFF \
    -D Trilinos_ENABLE_Komplex BOOL : OFF \
    -D Trilinos_ENABLE_TriKota BOOL : OFF \
    -D Trilinos_ENABLE_Moertel BOOL : OFF \
    -D Trilinos_ENABLE_Domi    BOOL : OFF \
    -D Trilinos_ENABLE_FEI     BOOL : OFF \
    \
    -D Trilinos_ENABLE_PyTrilinos BOOL : OFF \
    \
    -D Trilinos_ENABLE_Epetra BOOL : OFF \
    -S /opt/src/Trilinos -B /opt/build/Trilinos
ninja -j3