

TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/../LOCA_TestProblems)

TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)

IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_TPETRA AND NOX_ENABLE_LOCA)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_HouseholderBorderedSolve
    SOURCES
    HouseholderBorderedSolve.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

 TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_HouseholderTransposeBorderedSolve
    SOURCES
    HouseholderTransposeBorderedSolve.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_TransposeSolve
    SOURCES
    TransposeSolve.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_CompositeConstraint
    SOURCES
    CompositeConstraint.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_CompositeConstraintMVDX
    SOURCES
    CompositeConstraintMVDX.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_LOCA_NaturalContResidualFills
    SOURCES
    NaturalContResidualFills.cpp
    COMM serial mpi
    NUM_MPI_PROCS 2
    TESTONLYLIBS noxtestutils locatpetratestproblems
    ARGS -v
    PASS_REGULAR_EXPRESSION "All tests passed"
    )

ENDIF()
