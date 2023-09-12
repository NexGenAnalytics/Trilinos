
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)


# Dependency on Triutils fixes Bug 6449.
IF (${PACKAGE_NAME}_ENABLE_Triutils)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_BiCGStab_file
    SOURCES BiCGStabTpetraExFile.cpp
    COMM serial mpi
    ARGS "--verbose --max-iters=10"
         "--verbose --max-iters=10 --tol=1e-10"
         "--verbose --max-iters=10 --num-rhs=2"
    STANDARD_PASS_OUTPUT
    )

  TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_CopyExBiCGSTABFiles
    SOURCE_DIR ${${PACKAGE_NAME}_SOURCE_DIR}/tpetra/example/BiCGStab
    SOURCE_FILES cage4.hb
    EXEDEPS BiCGStab_file
    )

ENDIF()
