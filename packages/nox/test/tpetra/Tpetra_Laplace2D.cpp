#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Laplace2D.hpp"

// Laplace2D implementation
void Laplace2D::getMyNeighbours(const int i, const int nx, const int ny,
                  int &left, int &right,
                  int &lower, int &upper)
{
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

CRSM * Laplace2D::createLaplacian(const int nx, const int ny, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
  int numGlobalElements = nx * ny;
  int numLocalElements = 5; // TD: added
  // create a map
  Teuchos::RCP<const Map> map = Teuchos::rcp(new const Map(numGlobalElements, 5, 0, comm));
  // local number of rows
  numLocalElements = map->getLocalNumElements();
  // get update list
  Tpetra::global_size_t globalElements = map->getGlobalNumElements();

  double hx = 1.0 / (nx - 1);
  double hy = 1.0 / (ny - 1);
  double off_left = -1.0 / (hx * hx);
  double off_right = -1.0 / (hx * hx);
  double off_lower = -1.0 / (hy * hy);
  double off_upper = -1.0 / (hy * hy);
  double diag = 2.0 / (hx * hx) + 2.0 / (hy * hy);

  int left, right, lower, upper;

  // a bit overestimated the nonzero per row

  Teuchos::RCP<CRSM> A = Teuchos::rcp(new CRSM(map, 5));
  // Add  rows one-at-a-time

  double *values = new double[4];
  int *indices = new int[4];

  for (int i = 0; i < numMyElements; ++i)
  {
    int numEntries = 0;
    getMyNeighbours(globalElements[i], nx, ny, left, right, lower, upper);
    if (left != -1)
    {
      indices[numEntries] = left;
      values[numEntries] = off_left;
      ++numEntries;
    }
    if (right != -1)
    {
      indices[numEntries] = right;
      values[numEntries] = off_right;
      ++numEntries;
    }
    if (lower != -1)
    {
      indices[numEntries] = lower;
      values[numEntries] = off_lower;
      ++numEntries;
    }
    if (upper != -1)
    {
      indices[numEntries] = upper;
      values[numEntries] = off_upper;
      ++numEntries;
    }
    // put the off-diagonal entries
    a->insertGlobalValues(myGlobalElements[i], numEntries, values, indices);
    // Put in the diagonal entry
    a->insertGlobalValues(myGlobalElements[i], 1, &diag, myGlobalElements + i);
  }

  // put matrix in local ordering
  a->fillComplete();

  delete[] indices;
  delete[] values;
  delete map;

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
           const Teuchos::RCP<const Teuchos::Comm<int>> &comm) :
  nx_(nx), ny_(ny), lambda_(lambda)
{
  hx_ = 1.0/(nx_-1);
  hy_ = 1.0/(ny_-1);
  matrix_ = Laplace2D::createLaplacian(nx_,ny_,comm);
}

// destructor
PDEProblem::~PDEProblem()
{
  delete matrix_;
}

// compute F(x)
void PDEProblem::computeF( const TV & x, TV & f )
{
  // reset diagonal entries
  double diag =  2.0/(hx_*hx_) + 2.0/(hy_*hy_);

  int numMyElements = matrix_->getMap()->getLocalNumElements();
  // get update list
  int * myGlobalElements = matrix_->getMap()->getLocalElementList();

  for( int i = 0; i < numMyElements; ++i )
  {
    // Put in the diagonal entry
    matrix_->replaceGlobalValues(myGlobalElements[i], 1, &diag, myGlobalElements+i);
  }
  // matrix-vector product (intra-processes communication occurs in this call)
  matrix_->multiply( false, x, f );

  // add diagonal contributions
  for( int i = 0; i < numMyElements; ++i )
  {
    // Put in the diagonal entry
    f[i] += lambda_*exp(x[i]);
  }
}

// update the Jacobian matrix for a given x
void PDEProblem::updateJacobian( const TV & x )
{
  double diag = 2.0/(hx_*hx_) + 2.0/(hy_*hy_);

  int numMyElements = matrix_->getMap().numMyElements();
  // get update list
  int * myGlobalElements = matrix_->getMap().myGlobalElements();

  for( int i = 0; i < NumMyElements; ++i )
  {
    // Put in the diagonal entry
    double newdiag = diag + lambda_*exp(x[i]);
    matrix_->replaceGlobalValues(myGlobalElements[i], 1, &newdiag, myGlobalElements+i);
  }
}

TEUCHOS_UNIT_TEST(Tpetra_Laplace2D, Laplace2D)
{
  try {
    bool verbose = Teuchos::UnitTestRepository::verboseUnitTests();

    // Get the process ID and the total number of processors
    int myPID = comm.getRank();
#ifdef HAVE_MPI
    int numProc = comm.getSize();
#endif

  // define the parameters of the nonlinear PDE problem
  int nx = 5;
  int ny = 6;
  double lambda = 1.0;

  PDEProblem problem(nx,ny,lambda,&comm);

  // starting solution, here a zero vector
  TV initialGuess(problem.getMatrix()->getMap());
  initialGuess.putScalar(0.0);

  // random vector upon which to apply each operator being tested
  TV directionVec(problem.getMatrix()->getMap());
  directionVec.random();

  // Set up the problem interface
  Teuchos::RCP<SimpleProblemInterface> interface =
    Teuchos::rcp(new SimpleProblemInterface(&problem) );

  // Set up theolver options parameter list
  Teuchos::RCP<Teuchos::ParameterList> noxParamsPtr = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList & noxParams = *(noxParamsPtr.get());

  // Set the nonlinear solver method
  noxParams.set("Nonlinear Solver", "Line Search Based");

  // Set up the printing utilities
  // Only print output if the "-v" flag is set on the command line
  Teuchos::ParameterList& printParams = noxParams.sublist("Printing");
  printParams.set("MyPID", myPID);
  printParams.set("Output Precision", 5);
  printParams.set("Output Processor", 0);
  if( verbose )
    printParams.set("Output Information",
        NOX::Utils::OuterIteration +
        NOX::Utils::OuterIterationStatusTest +
        NOX::Utils::InnerIteration +
        NOX::Utils::Parameters +
        NOX::Utils::Details +
        NOX::Utils::Warning +
        NOX::Utils::TestDetails);
  else
    printParams.set("Output Information", NOX::Utils::Error +
        NOX::Utils::TestDetails);

  NOX::Utils printing(printParams);

  // Identify the test problem
  if (printing.isPrintType(NOX::Utils::TestDetails))
    printing.out() << "Starting epetra/NOX_Operators/NOX_Operators.exe" << std::endl;

  // Identify processor information
#ifdef HAVE_MPI
  if (printing.isPrintType(NOX::Utils::TestDetails)) {
    printing.out() << "Parallel Run" << std::endl;
    printing.out() << "Number of processors = " << numProc << std::endl;
    printing.out() << "Print Process = " << myPID << std::endl;
  }
  comm.barrier();
  if (printing.isPrintType(NOX::Utils::TestDetails))
    printing.out() << "Process " << myPID << " is alive!" << std::endl;
  comm.barrier();
#else
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Serial Run" << std::endl;
#endif

    int status = 0;

    Teuchos::RCP<NOX::Tpetra::Interface::Required> iReq = interface;

    // Need a NOX::Epetra::Vector for constructor
    NOX::Epetra::Vector noxInitGuess(initialGuess, NOX::DeepCopy);

    // Analytic matrix
    Teuchos::RCP<CRSM> A = Teuchos::rcp( problem.getMatrix(), false );

    TV A_resultVec(problem.getMatrix()->getMap());
    interface->computeJacobian( initialGuess, *A );
    A->Apply( directionVec, A_resultVec );

    // FD operator
    Teuchos::RCP<CSRG> graph = Teuchos::rcp( const_cast<CSRG*>(&A->getGraph()), false );
    Teuchos::RCP<NOX::Epetra::FiniteDifference> FD = Teuchos::rcp(
      new NOX::Epetra::FiniteDifference(printParams, iReq, noxInitGuess, graph) );

    TV FD_resultVec(problem.getMatrix()->getMap());
    FD->computeJacobian(initialGuess, *FD);
    FD->Apply( directionVec, FD_resultVec );

    // Matrix-Free operator
    Teuchos::RCP<NOX::Epetra::MatrixFree> MF = Teuchos::rcp(
      new NOX::Epetra::MatrixFree(printParams, iReq, noxInitGuess) );

    TV MF_resultVec(problem.getMatrix()->getMap());
    MF->computeJacobian(initialGuess, *MF);
    MF->apply( directionVec, MF_resultVec );

    // Need NOX::Epetra::Vectors for tests
    NOX::Epetra::Vector noxAvec ( A_resultVec , NOX::DeepCopy );
    NOX::Epetra::Vector noxFDvec( FD_resultVec, NOX::DeepCopy );
    NOX::Epetra::Vector noxMFvec( MF_resultVec, NOX::DeepCopy );

    // Create a TestCompare class
    NOX::Epetra::TestCompare tester( printing.out(), printing);
    double abstol = 1.e-4;
    double reltol = 1.e-4 ;
    //NOX::TestCompare::CompareType aComp = NOX::TestCompare::Absolute;

    status += tester.testVector( noxFDvec, noxAvec, reltol, abstol,
                                "Finite-Difference Operator Apply Test" );
    status += tester.testVector( noxMFvec, noxAvec, reltol, abstol,
                                "Matrix-Free Operator Apply Test" );

    success = status==0;

    // Summarize test results
    if(success)
      printing.out() << "Test passed!" << std::endl;
    else
      printing.out() << "Test failed!" << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return -1;
}
