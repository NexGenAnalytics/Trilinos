//@HEADER
// TOREDO

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "BelosBlockGmresSolMgr.hpp"

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MatrixIO.hpp>

// Teuchos
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

template<class ScalarType>
int run(int argc, char *argv[]){
  using ST = ScalarType;
  using SCT = typename Teuchos::ScalarTraits<ST>;
  using MT = typename SCT::magnitudeType;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;
  
  using OP = typename Tpetra::Operator<ST,LO,GO,NT>;
  using MV = typename Tpetra::MultiVector<ST,LO,GO,NT>;
  using MVT = typename Belos::MultiVecTraits<ST,MV>;
  using OPT = typename Belos::OperatorTraits<ST,MV,OP>;

  using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;
  using tmultivector_t = Tpetra::MultiVector<ST,LO,GO,NT>;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  Teuchos::GlobalMPISession session(&argc, &argv, NULL);
  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    
  bool success = false;
  bool verbose = false;

  try {
    int MyPID = rank(*comm);
    //
    // Get test parameters from command-line processor
    //
    bool proc_verbose = false;
    bool leftprec = true;      // left preconditioning or right.
    int frequency = -1;        // how often residuals are printed by solver
    int numrhs = 5;            // total number of right-hand sides to solve for
    int maxiters = -1;         // maximum number of iterations for the solver to use
    std::string filename("bcsstk14.hb");
    double tol = 1.0e-5;       // relative residual tolerance

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("left-prec","right-prec",&leftprec,"Left preconditioning or right.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("filename",&filename,"Filename for Harwell-Boeing test matrix.");
    cmdp.setOption("tol",&tol,"Relative residual tolerance used by Minres solver.");
    cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 := adapted to problem/block size).");

    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    if (!verbose)
      frequency = -1;  // Reset frequency if verbosity is off

    proc_verbose = ( verbose && (MyPID==0) );
    if (proc_verbose) {
      std::cout << Belos::Belos_Version() << std::endl << std::endl;
    }

    //
    // Get the problem
    //
    RCP<tcrsmatrix_t> A;
    Tpetra::Utils::readHBMatrix(filename,comm,A);
    RCP<const Tpetra::Map<> > map = A->getDomainMap();

    // Create initial vectors
    RCP<tmultivector_t> X, B;
    X = rcp(new MV(map, numrhs));
    MVT::MvRandom( *X );
    B = rcp(new MV(map, numrhs));
    OPT::Apply(*A, *X, *B);
    MVT::MvInit(*X, 0.0);
    
    //
    // *****Create parameter list for the Minres solver manager*****
    //
    const int NumGlobalElements = B->getGlobalLength();
    if (maxiters == -1)
      maxiters = NumGlobalElements - 1; // maximum number of iterations to run
    //
    ParameterList belosList;
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    belosList.set( "Flexible Gmres", true );               // use FGMRES
                                                           
    int verbLevel = Belos::Errors + Belos::Warnings;
    if (verbose) {
      verbLevel += Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails;
    }
    belosList.set( "Verbosity", verbLevel );
    if (verbose) {
      if (frequency > 0) {
        belosList.set( "Output Frequency", frequency );
      }
    }

    // Set parameters for the inner GMRES (preconditioning) iteration.
    ParameterList innerBelosList;
    innerBelosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    innerBelosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    innerBelosList.set( "Verbosity", Belos::Errors + Belos::Warnings );

    // *****Construct linear problem for the inner iteration using A *****
    Belos::LinearProblem<ST,MV,OP> innerProblem;
    innerProblem.setOperator( A );
    innerProblem.setLabel( "Belos Preconditioner Solve" );

    //  
    // *****Create the inner block Gmres iteration********
    //  
    RCP<Belos::TpetraOperator<ST>> innerSolver;
    innerSolver = rcp( new Belos::TpetraOperator<ST>( rcpFromRef(innerProblem), rcpFromRef(innerBelosList), "Block Gmres", true ) );

    //
    // Construct a linear problem instance with GMRES as preconditoner.
    //
    Belos::LinearProblem<ST,MV,OP> problem( A, X, B );
    problem.setInitResVec(B);
    problem.setRightPrec(innerSolver);
    problem.setLabel( "Belos Flexible Gmres Solve" );
    bool set = problem.setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return -1;
    }
    //
    // *******************************************************************
    // *************Start the block Gmres iteration***********************
    // *******************************************************************
    //
    Belos::BlockGmresSolMgr<ST,MV,OP> solver( rcpFromRef(problem), rcpFromRef(belosList) );
    
    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Max number of Minres iterations: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    //
    // Perform solve
    //
    Belos::ReturnType ret = solver.solve();
    //
    // Compute actual residuals.
    //
    bool badRes = false;
    std::vector<MT> actual_resids( numrhs );
    std::vector<MT> rhs_norm( numrhs );
    MV resid(map, numrhs);
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

    success = ret==Belos::Converged && !badRes;

    if (success) {
      if (proc_verbose)
        std::cout << std::endl << "End Result: TEST PASSED" << std::endl;
    } else {
      if (proc_verbose)
        std::cout << std::endl << "End Result: TEST FAILED" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // run with different scalar types
  run<double>(argc, argv);
  // run<float>(argc, argv);
}
