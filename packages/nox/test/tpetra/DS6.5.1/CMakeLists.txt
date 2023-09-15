

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_TPETRA)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    DS6_5_1
    SOURCES
    DennisSchnabelTpetra.cpp
    DennisSchnabelTpetra.hpp
    Example_Tpetra.cpp
    Problem_Interface_Tpetra.cpp
    Problem_Interface_Tpetra.hpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    PASS_REGULAR_EXPRESSION "Test passed!"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    DS6_5_1_TrustRegion
    SOURCES
    DennisSchnabelTpetra.cpp
    DennisSchnabelTpetra.hpp
    Example_TrustRegionTpetra.cpp
    Problem_Interface_Tpetra.cpp
    Problem_Interface_Tpetra.hpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    PASS_REGULAR_EXPRESSION "Test passed!"
    )

ENDIF()
