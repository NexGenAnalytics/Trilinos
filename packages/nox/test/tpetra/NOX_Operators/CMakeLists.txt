ADD_SUBDIRECTORY(src)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  NOX_Operators_Tpetra
  SOURCES
  NOX_Tpetra_Interface_Jacobian.hpp # this file epetra equivalent was in src-epetra. move tpetra one to src-tpetra ?
  NOX_Tpetra_Interface_Preconditioner.hpp # this file epetra equivalent was in src-epetra. move tpetra one to src-tpetra ?
  NOX_Tpetra_Interface_Required.hpp # this file epetra equivalent was in src-epetra. move tpetra one to src-tpetra ?
  Laplace2D_Tpetra.hpp
  Laplace2D_Tpetra.cpp
  test.cpp
  # TESTONLYLIBS noxtestutils
  COMM serial mpi
  NUM_MPI_PROCS 1
  # PASS_REGULAR_EXPRESSION "Test passed!"
  )

