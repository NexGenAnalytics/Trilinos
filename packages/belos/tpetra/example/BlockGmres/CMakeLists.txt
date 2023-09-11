
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Galeri)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)

IF (${PACKAGE_NAME}_ENABLE_Triutils)

  TRIBITS_ADD_EXECUTABLE(
    BlockGmres_Tpetra_File_Ex
    SOURCES BlockGmresTpetraExFile.cpp
    COMM serial mpi
    )

  TRIBITS_ADD_EXECUTABLE(
    PseudoBlockGmres_Tpetra_File_Ex
    SOURCES PseudoBlockGmresTpetraExFile.cpp
    COMM serial mpi
    )

  TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_CopyExampleBlockGmresFiles
    SOURCE_DIR ${Belos_SOURCE_DIR}/tpetra/example/BlockGmres
    SOURCE_FILES orsirr1.hb identity.mtx
    EXEDEPS BlockGmres_Tpetra_File_Ex
    )

  ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Ifpack)
  IF(${PACKAGE_NAME}_ENABLE_Ifpack)

    TRIBITS_ADD_EXECUTABLE(
      BlockPrecGmres_Tpetra_File_Ex
      SOURCES BlockPrecGmresTpetraExFile.cpp
      COMM serial mpi
      )

    TRIBITS_ADD_EXECUTABLE(
      PseudoBlockPrecGmres_Tpetra_File_Ex
      SOURCES PseudoBlockPrecGmresTpetraExFile.cpp
      COMM serial mpi
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      BlockFlexPrecGmres_Tpetra_File_Ex
      SOURCES BlockFlexGmresTpetraExFile.cpp
      COMM serial mpi
      ARGS "--verbose --filename=orsirr1.hb"
           "--verbose --filename=identity.mtx --use-rhs=true"
      STANDARD_PASS_OUTPUT
      )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      BlockGmresPoly_Tpetra_File_Ex
      SOURCES BlockGmresPolyTpetraExFile.cpp
      COMM serial mpi
      ARGS "--verbose --filename=orsirr1.hb --precond=none --max-degree=5 --poly-type=Arnoldi"
           "--verbose --filename=orsirr1.hb --precond=none --max-degree=5 --poly-type=Gmres"
           "--verbose --filename=orsirr1.hb --precond=none --max-degree=5 --poly-type=Roots"
      )

  ENDIF(${PACKAGE_NAME}_ENABLE_Ifpack)

ENDIF(${PACKAGE_NAME}_ENABLE_Triutils AND ${PACKAGE_NAME}_ENABLE_TpetraExt)

IF (${PACKAGE_NAME}_ENABLE_Galeri)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    BlockGmres_Galeri_Ex
    SOURCES BlockGmresTpetraGaleriEx.cpp
    COMM serial mpi
    ARGS "--verbose --nx=10 --ortho=DGKS"
         "--verbose --nx=10 --ortho=ICGS"
         "--verbose --nx=10 --ortho=IMGS"
    STANDARD_PASS_OUTPUT
    )

  ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Ifpack)
  IF(${PACKAGE_NAME}_ENABLE_Ifpack)

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      BlockPrecGmres_Galeri_Ex
      SOURCES BlockPrecGmresTpetraGaleriEx.cpp
      COMM serial mpi
      ARGS "--verbose --nx=10 --ortho=DGKS"
           "--verbose --nx=10 --ortho=ICGS"
           "--verbose --nx=10 --ortho=IMGS"
      STANDARD_PASS_OUTPUT
      )

  ENDIF(${PACKAGE_NAME}_ENABLE_Ifpack)

ENDIF (${PACKAGE_NAME}_ENABLE_Galeri)

