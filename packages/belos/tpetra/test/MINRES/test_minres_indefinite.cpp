//@HEADER
// TOREDO

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosMinresSolMgr.hpp"

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

int main(int argc, char *argv[])
{
  using SC = typename Tpetra::Vector<double>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using SCT = typename Teuchos::ScalarTraits<SC>;
  using MT = typename SCT::magnitudeType;
  using MV = typename Tpetra::MultiVector<SC,LO,GO,NT>;
  using OP = typename Tpetra::Operator<SC,LO,GO,NT>;
  using MVT = typename Belos::MultiVecTraits<SC,MV>;
  using OPT = typename Belos::OperatorTraits<SC,MV,OP>;
 
  using tcrsmatrix_t = Tpetra::CrsMatrix<SC,LO,GO,NT>;
  using tmap_t = Tpetra::Map<LO,GO,NT>;
  using tmultivector_t = Tpetra::MultiVector<SC,LO,GO,NT>;

  using Teuchos::CommandLineProcessor;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // AM: TODO Better initialize MPI and get communicator
  MPI_Init(&argc,&argv);
  const auto comm = Tpetra::getDefaultComm();
  const int MyPID = 0;

  bool success = true;
  bool verbose = false;
  bool debug = false;
  
  try {
    int frequency = -1;        // frequency of status test output.
    int numrhs = 1;            // number of right-hand sides to solve for
    int maxiters = -1;         // maximum number of iterations allowed per linear system
    MT tol = 1.0e-8;           // relative residual tolerance

    // Define command-line arguments
    CommandLineProcessor cmdp(false,true);
    cmdp.setOption ("verbose", "quiet", &verbose, "Print messages and results.");
    cmdp.setOption ("debug", "nodebug", &debug,
        "Print debugging information from the solver.");
    cmdp.setOption ("frequency", &frequency,
        "Solver's frequency for printing residuals (#iters).");
    cmdp.setOption ("tol", &tol,
        "Relative residual tolerance used by MINRES solver.");
    cmdp.setOption ("num-rhs", &numrhs,
        "Number of right-hand sides for which to solve (> 0).");
    cmdp.setOption ("max-iters", &maxiters,
        "Maximum number of iterations per linear system.  -1 means "
        "we choose, based on problem size.");

    // Parse command-line arguments and fetch values
    if (cmdp.parse(argc,argv) != CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      std::cout << "Failed to parse command-line arguments!" << std::endl;
      std::cout << "End Result: TEST FAILED" << std::endl;
      return -1;
    }

    // **********************************************************************
    // ******************Set up the problem to be solved*********************
    // construct diagonal matrix
    const int NumGlobalElements = 100;
    const int m = 4; // number of negative eigenvalues

    // Create diagonal matrix with n-m positive and m negative eigenvalues.
    RCP<tmap_t> Map = rcp(new tmap_t(NumGlobalElements, 0, comm));

    RCP<tcrsmatrix_t> A = rcp( new tcrsmatrix_t(Map, 1));
    for ( size_t k=0; k<Map->getLocalNumElements(); k++ )
    {
      int GIDk = Map->getGlobalElement(k);
      double val = 2*(GIDk-m) + 1;
      A->insertGlobalValues( GIDk, 1, &val, &GIDk );
    }
    A->fillComplete();
    TEUCHOS_ASSERT_EQUALITY( true, A->isFillComplete() );
    TEUCHOS_ASSERT_EQUALITY( true, A->isStorageOptimized() );

    //
    // Make some (multi)vectors, for use in testing: an exact solution,
    // its corresponding right-hand side, and an initial guess.
    //

    // Make a random exact solution.
    RCP<tmultivector_t> X_exact (new tmultivector_t (Map, numrhs));
    MVT::MvRandom (*X_exact);

    // Compute the right-hand side as B = A*X.
    RCP<tmultivector_t> B = MVT::Clone (*X_exact, numrhs);
    OPT::Apply (*A, *X_exact, *B);

    // Choose an initial guess of all zeros.
    RCP<tmultivector_t> X = MVT::Clone (*X_exact, numrhs);
    MVT::MvInit (*X, SC(0.0));

    // **********************************************************************
    //
    // Set parameters that the user asked us to pick intelligently.
    //

    // Maximum number of iterations: set according to problem size.
    // maxiters=numGlobalElements-1 may not always converge, so we
    // set to numGlobalElements+1 for good measure.
    if (maxiters == -1)
      maxiters = NumGlobalElements + 1;

    // In a nonverbose test, the frequency should be set to -1, which
    // Belos interprets as "no intermediate status output."  Override
    // whatever the user may have specified.
    if (! verbose)
      frequency = -1;
    // Silently fix a bad frequency value.
    else if (frequency < 0 && frequency != -1)
      frequency = -1;

    // Validate command-line arguments
    TEUCHOS_TEST_FOR_EXCEPTION( tol < 0, std::invalid_argument,
        "Relative residual tolerance must be nonnegative, but "
        "you supplied tol = " << tol << "." );
    TEUCHOS_TEST_FOR_EXCEPTION( numrhs < 1, std::invalid_argument,
        "MINRES test requires at least one right-hand side, but "
        "you set the number of right-hand sides to "
        << numrhs << "." );
    TEUCHOS_TEST_FOR_EXCEPTION( maxiters < 1, std::invalid_argument,
        "MINRES test requires at least one iteration, but you "
        "set the maximum number of iterations to "
        << maxiters << "." );

    const bool proc_verbose = verbose && (MyPID==0);  /* Only print on the zero processor */

    //
    // ********Other information used by block solver***********
    // *****************(can be user specified)******************
    //

    //
    ParameterList belosList;
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested

    int verbosity = Belos::Errors + Belos::Warnings;
    if (verbose) {
      verbosity += Belos::TimingDetails + Belos::StatusTestDetails;
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    if (debug) {
      verbosity += Belos::Debug;
    }
    belosList.set( "Verbosity", (int) verbosity );
    //
    // Construct an unpreconditioned linear problem instance.
    //
    Belos::LinearProblem< SC, MV, OP > problem (A, X, B);
    {
      const bool set = problem.setProblem();
      TEUCHOS_TEST_FOR_EXCEPTION( set == false, std::logic_error,
          "Belos::LinearProblem failed to set up correctly (setP"
          "roblem() returned false)!  This probably means we imp"
          "lemented our test incorrectly." );
    }
    //
    // *******************************************************************
    // ****************Start the MINRES iteration*************************
    // *******************************************************************
    //
    // Create an iterative solver manager.
    RCP< Belos::SolverManager<SC,MV,OP> > newSolver
      = rcp( new Belos::MinresSolMgr<SC,MV,OP>(rcp(&problem,false), rcp(&belosList,false)));

    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    //
    // Perform solve
    //
    Belos::ReturnType ret = Belos::Unconverged;
    try {
      ret = newSolver->solve();
    } catch (std::exception& e) {
      if (MyPID == 0)
        std::cout << "MINRES solver threw an exception: " << e.what() << std::endl;
      throw e;
    }
    //
    // Compute actual residuals.
    //
    bool badRes = false;
    std::vector<double> actual_resids( numrhs );
    std::vector<double> rhs_norm( numrhs );
    tmultivector_t resid(A->getMap(), numrhs);
    OPT::Apply( *A, *X, resid );
    MVT::MvAddMv( -1.0, resid, 1.0, *B, resid );
    MVT::MvNorm( resid, actual_resids );
    MVT::MvNorm( *B, rhs_norm );
    if (proc_verbose) {
      std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
      for ( int i=0; i<numrhs; i++) {
        double actRes = actual_resids[i]/rhs_norm[i];
        std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
        if (actRes > tol) badRes = true;
      }
    }

    if (ret!=Belos::Converged || badRes) {
      success = false;
      if (proc_verbose)
        std::cout << std::endl << "End Result: TEST FAILED" << std::endl;
    } else {
      success = true;
      if (proc_verbose)
        std::cout << std::endl << "End Result: TEST PASSED" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  MPI_Finalize();

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
