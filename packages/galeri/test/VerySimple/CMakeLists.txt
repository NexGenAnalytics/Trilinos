
IF (${PACKAGE_NAME}_ENABLE_Epetra)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    VerySimple
    SOURCES cxx_main.cpp
    ARGS -v
    COMM serial
    )

ENDIF()

IF (${PACKAGE_NAME}_ENABLE_Tpetra)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    VerySimple_Tpetra
    SOURCES cxx_main_tpetra.cpp
    ARGS -v
    COMM serial mpi
    )

ENDIF()
