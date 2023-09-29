// Teuchos
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

// Tpetra
#include <Tpetra_Core.hpp>

// Lib headers
#include "Pitchfork_FiniteElementProblem.hpp"
#include "LOCALinearConstraint.hpp"

// This is a temporary test created to run any code from the test libs
// to checkk if it is running
int main(int argc, char* argv[]) {

  using ST = typename Tpetra::MultiVector<double>::scalar_type;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  auto comm = Tpetra::getDefaultComm();
  const int myRank = comm->getRank();

  // test Pitchfork_FiniteElementProblem (inheriting FiniteElementProblem)
  int numGlobalElements = 1000;
  auto feProblem = Pitchfork_FiniteElementProblem<ST>(numGlobalElements, comm);

  if (myRank == 0) {
    std::cout << "End Result: TEST PASSED" << std::endl;
  }
}