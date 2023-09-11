
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)

IF (${PACKAGE_NAME}_ENABLE_Triutils)

  TRIBITS_ADD_EXECUTABLE(
    BlockGmres_Tpetra_File_Ex
    SOURCES BlockGmresTpetraExFile.cpp
    COMM serial mpi
    )

  TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_CopyExampleBlockGmresFiles
    SOURCE_DIR ${Belos_SOURCE_DIR}/tpetra/example/BlockGmres
    SOURCE_FILES orsirr1.hb identity.mtx
    EXEDEPS BlockGmres_Tpetra_File_Ex
    )

ENDIF(${PACKAGE_NAME}_ENABLE_Triutils)
