#include "Tpetra_Laplace2D.hpp"
#include "NOX_Utils.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"

#include <Tpetra_CrsMatrix_def.hpp>
#include <Tpetra_Map_def.hpp>
#include <Tpetra_Vector_def.hpp>

#include <array>

#if defined HAVE_TPETRACORE_CUDA
#define NUM_LOCAL 10000
#else
#define NUM_LOCAL 100
#endif // HAVE_TPETRACORE_CUDA
constexpr std::size_t numLocalElements = NUM_LOCAL;

#ifndef HAVE_MPI
static constexpr bool NOX_HAVE_MPI = false;
#else
static constexpr bool NOX_HAVE_MPI = true;
#endif // HAVE_MPI

// Laplace2D implementation
void Laplace2D::getMyNeighbours(const NOX::GlobalOrdinal i,
                                const NOX::GlobalOrdinal nx,
                                const NOX::GlobalOrdinal ny,
                                NOX::GlobalOrdinal &left,
                                NOX::GlobalOrdinal &right,
                                NOX::GlobalOrdinal &lower,
                                NOX::GlobalOrdinal &upper) {
  int ix, iy;
  ix = i % nx;
  iy = (i - ix) / nx;

  if (ix == 0)
    left = -1;
  else
    left = i - 1;
  if (ix == nx - 1)
    right = -1;
  else
    right = i + 1;
  if (iy == 0)
    lower = -1;
  else
    lower = i - nx;
  if (iy == ny - 1)
    upper = -1;
  else
    upper = i + nx;

  return;
}

Teuchos::RCP<NOX::TCrsMatrix>
Laplace2D::createLaplacian(const int nx, const int ny,
                           const Teuchos::RCP<const Teuchos::Comm<int>> &comm) {
  int numGlobalElements = nx * ny;
  int numLocalElements = 5;

  // Create Tpetra vectors
  auto map =
      Teuchos::rcp(new NOX::TMap(numGlobalElements, numLocalElements, 0, comm));

  // get update list
  const auto myGlobalElements = map->getMyGlobalIndices();

  NOX::Scalar hx = 1.0 / (nx - 1);
  NOX::Scalar hy = 1.0 / (ny - 1);
  NOX::Scalar off_left = -1.0 / (hx * hx);
  NOX::Scalar off_right = -1.0 / (hx * hx);
  NOX::Scalar off_lower = -1.0 / (hy * hy);
  NOX::Scalar off_upper = -1.0 / (hy * hy);
  NOX::Scalar diag = 2.0 / (hx * hx) + 2.0 / (hy * hy);

  NOX::GlobalOrdinal left, right, lower, upper;

  // a bit overestimated the nonzero per row
  auto a = Teuchos::rcp(new NOX::TCrsMatrix(map, numLocalElements));

  std::array<NOX::Scalar, 4> values;
  std::array<NOX::GlobalOrdinal, 4> indices;

  for (NOX::LocalOrdinal i = 0; i < numLocalElements; ++i) {
    int numEntries = 0;
    getMyNeighbours(myGlobalElements[i], nx, ny, left, right, lower, upper);
    if (left != -1) {
      indices[numEntries] = left;
      values[numEntries] = off_left;
      ++numEntries;
    }
    if (right != -1) {
      indices[numEntries] = right;
      values[numEntries] = off_right;
      ++numEntries;
    }
    if (lower != -1) {
      indices[numEntries] = lower;
      values[numEntries] = off_lower;
      ++numEntries;
    }
    if (upper != -1) {
      indices[numEntries] = upper;
      values[numEntries] = off_upper;
      ++numEntries;
    }
    // put the off-diagonal entries
    a->insertGlobalValues(myGlobalElements[i], numEntries, values.data(),
                          indices.data());
    // Put in the diagonal entry
    a->insertGlobalValues(myGlobalElements[i], 1, &diag,
                          myGlobalElements.data() + i);
  }

  // put matrix in local ordering
  a->fillComplete();

  return a;

} /* createJacobian */

// ==========================================================================
// This class contians the main definition of the nonlinear problem at
// hand. A method is provided to compute F(x) for a given x, and another
// method to update the entries of the Jacobian matrix, for a given x.
// As the Jacobian matrix J can be written as
//    J = L + diag(lambda*exp(x[i])),
// where L corresponds to the discretization of a Laplacian, and diag
// is a diagonal matrix with lambda*exp(x[i]). Basically, to update
// the jacobian we simply update the diagonal entries. Similarly, to compute
// F(x), we reset J to be equal to L, then we multiply it by the
// (distributed) vector x, then we add the diagonal contribution
// ==========================================================================

// constructor. Requires the number of nodes along the x-axis
// and y-axis, the value of lambda, and the communicator
// (to define a Map, which is a linear map in this case)
PDEProblem::PDEProblem(const int nx, const int ny, const double lambda,
                       const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : nx_(nx), ny_(ny), lambda_(lambda) {
  hx_ = 1.0 / (nx_ - 1);
  hy_ = 1.0 / (ny_ - 1);
  matrix_ = Laplace2D::createLaplacian(nx_, ny_, comm);
}

// compute F(x)
void PDEProblem::computeF(const NOX::TVector &x, NOX::TVector &f) {
  // reset diagonal entries
  double diag = 2.0 / (hx_ * hx_) + 2.0 / (hy_ * hy_);

  int numMyElements = matrix_->getMap()->getLocalNumElements();
  // get update list
  auto myGlobalElements = matrix_->getMap()->getLocalElementList();

  for (int i = 0; i < numMyElements; ++i) {
    // Put in the diagonal entry
    matrix_->replaceGlobalValues(myGlobalElements[i], 1, &diag,
                                 myGlobalElements.data() + i);
  }
  // matrix-vector product (intra-processes communication occurs in this call)
  matrix_->apply(x, f);

  auto fValues = f.getLocalViewHost(Tpetra::Access::ReadWrite);
  auto xValues = x.getLocalViewHost(Tpetra::Access::ReadOnly);

  // add diagonal contributions (extent(1)?)
  for (size_t i = 0; i < fValues.extent(1); ++i) {
    // Put in the diagonal entry
    fValues(0, i) += lambda_ * exp(xValues(0, i));
  }
}

// update the Jacobian matrix for a given x
void PDEProblem::updateJacobian(const NOX::TVector &x) {
  double diag = 2.0 / (hx_ * hx_) + 2.0 / (hy_ * hy_);

  auto numMyElements = matrix_->getMap()->getLocalNumElements();
  // get update list
  auto myGlobalElements = matrix_->getMap()->getLocalElementList();

  auto xValues = x.getLocalViewHost(Tpetra::Access::ReadOnly);

  for (size_t i = 0; i < numMyElements; ++i) {
    // Put in the diagonal entry
    double newdiag = diag + lambda_ * exp(xValues(0, i));
    matrix_->replaceGlobalValues(myGlobalElements[i], 1, &newdiag,
                                 myGlobalElements.data() + i);
  }
}

TEUCHOS_UNIT_TEST(Tpetra_Laplace2D, Laplace2D) {
  bool verbose = Teuchos::UnitTestRepository::verboseUnitTests();

  auto comm = Tpetra::getDefaultComm();

  // Get the process ID and the total number of processors
  int myPID = comm->getRank();
  int numProc = comm->getSize();

  // define the parameters of the nonlinear PDE problem
  int nx = 5;
  int ny = 6;
  double lambda = 1.0;

  PDEProblem problem(nx, ny, lambda, comm);

  // starting solution, here a zero vector
  NOX::TVector initialGuess(problem.getMatrix()->getMap());
  initialGuess.putScalar(0.0);

  // random vector upon which to apply each operator being tested
  NOX::TVector directionVec(problem.getMatrix()->getMap());
  directionVec.randomize();

  // Set up the problem interface
  auto interface = Teuchos::rcp(new SimpleProblemInterface(&problem));

  // Set up theolver options parameter list
  auto noxParamsPtr = Teuchos::rcp(new Teuchos::ParameterList);
  auto &noxParams = *(noxParamsPtr.get());

  // Set the nonlinear solver method
  noxParams.set("Nonlinear Solver", "Line Search Based");

  // Set up the printing utilities
  // Only print output if the "-v" flag is set on the command line
  Teuchos::ParameterList &printParams = noxParams.sublist("Printing");
  printParams.set("MyPID", myPID);
  printParams.set("Output Precision", 5);
  printParams.set("Output Processor", 0);
  if (verbose)
    printParams.set("Output Information",
                    NOX::Utils::OuterIteration +
                        NOX::Utils::OuterIterationStatusTest +
                        NOX::Utils::InnerIteration + NOX::Utils::Parameters +
                        NOX::Utils::Details + NOX::Utils::Warning +
                        NOX::Utils::TestDetails);
  else
    printParams.set("Output Information",
                    NOX::Utils::Error + NOX::Utils::TestDetails);

  NOX::Utils printing(printParams);

  // Identify the test problem
  if (printing.isPrintType(NOX::Utils::TestDetails))
    printing.out() << "Starting tpetra/NOX_Operators/NOX_Operators.exe"
                   << std::endl;

  // Identify processor information
  if constexpr (NOX_HAVE_MPI) {
    if (printing.isPrintType(NOX::Utils::TestDetails)) {
      printing.out() << "Parallel Run" << std::endl;
      printing.out() << "Number of processors = " << numProc << std::endl;
      printing.out() << "Print Process = " << myPID << std::endl;
    }
    comm->barrier();
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Process " << myPID << " is alive!" << std::endl;
    comm->barrier();
  } else {
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Serial Run" << std::endl;
  }

  int status = 0;

  auto iReq = interface;

  // Analytic matrix
  auto A = problem.getMatrix();

  NOX::TVector A_resultVec(problem.getMatrix()->getMap());
  interface->computeJacobian(initialGuess, *A);
  A->apply(directionVec, A_resultVec);

  // FD operator
  auto graph = A->getGraph();
  auto FD = Teuchos::rcp(new NOX::Tpetra::FiniteDifference(
      printParams, iReq, noxInitGuess, graph));

  NOX::TVector FD_resultVec(problem.getMatrix()->getMap());
  FD->computeJacobian(initialGuess, *FD);
  FD->apply(directionVec, FD_resultVec);

  // // Matrix-Free operator
  // auto MF = Teuchos::rcp( new NOX::Epetra::MatrixFree(printParams, iReq,
  // noxInitGuess));

  // NOX::TVector MF_resultVec(problem.getMatrix()->getMap());
  // MF->computeJacobian(initialGuess, *MF);
  // MF->apply(directionVec, MF_resultVec);

  // // Need NOX::Epetra::Vectors for tests
  // NOX::Epetra::Vector noxAvec(A_resultVec, NOX::DeepCopy);
  // NOX::Epetra::Vector noxFDvec(FD_resultVec, NOX::DeepCopy);
  // NOX::Epetra::Vector noxMFvec(MF_resultVec, NOX::DeepCopy);

  // // Create a TestCompare class
  // NOX::TestCompare tester(printing.out(), printing);
  // double abstol = 1.e-4;
  // double reltol = 1.e-4;
  // // NOX::TestCompare::CompareType aComp = NOX::TestCompare::Absolute;

  // status += tester.testVector(noxFDvec, noxAvec, reltol, abstol,
  //                             "Finite-Difference Operator Apply Test");
  // status += tester.testVector(noxMFvec, noxAvec, reltol, abstol,
  //                             "Matrix-Free Operator Apply Test");

  // success = status == 0;

  // Summarize test results
  if (success)
    printing.out() << "Test passed!" << std::endl;
  else
    printing.out() << "Test failed!" << std::endl;
}
