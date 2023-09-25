
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_GeneralizedDavidson_solvertest
  SOURCES cxx_main_solvertest.cpp
  ARGS
    "--verbose"
#    "--debug"
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_GeneralizedDavidson_test
  SOURCES cxx_main.cpp
  ARGS
    "--verbose"
#    "--debug"
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_GeneralizedDavidson_nh_test
  SOURCES cxx_main_nh.cpp
  ARGS
    "--verbose"
#    "--debug"
  COMM serial mpi
  )
