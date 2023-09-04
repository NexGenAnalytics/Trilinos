//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

// Purpose
// The example tests the successive right-hand sides capabilities of ML
// and Belos on a heat flow u_t = u_xx problem.
//
// A sequence of linear systems with the same coefficient matrix and
// different right-hand sides is solved.  A seed space is generated dynamically,
// and a deflated linear system is solved.  After each solves, the first
// few Krylov vectors are saved, and used to reduce the number of iterations
// for later solves.
// The optimal numbers of vectors to deflate and save are not known.
// Presently, the maximum number of vectors to deflate (seed space dimension)
// and to save are user paraemters.
// The seed space dimension is less than or equal to total number of vectors saved.
// The difference between the seed space dimension and the total number of vectors,
// is the number of vectors used to update the seed space after each solve.
// I guess that a seed space whose dimension is a small fraction of the total space
// will be best.
//
// maxSave=1 and maxDeflate=0 uses no recycling (not tested ).
//
// TODO: Instrument with timers, so that we can tell what is going on besides
//       by counting the numbers of iterations.
//
//
// \author David M. Day
//
// \data Last modified 2007 December 11

// #include "ml_include.h" //--enable-epetra --enable-teuchos.

#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "TpetraExt_MatrixMatrix.hpp"
// #include "MueLu_CreateTpetraPreconditioner.hpp"
//#include "ml_MultiLevelPreconditioner.h" // ML

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPCPGSolMgr.hpp"
#include "BelosTpetraTestFramework.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

int main(int argc, char *argv[]) {
  //
  // Laplace's equation, homogeneous Dirichlet boundary conditions, [0,1]^2
  // regular mesh, Q1 finite elements
  //

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using std::cout;
  using std::endl;

  typedef Tpetra::MultiVector<>::scalar_type     Scalar;
  typedef Teuchos::ScalarTraits<Scalar>          SCT;
  typedef SCT::magnitudeType                     MT;
  typedef Tpetra::Map<>::local_ordinal_type      LO;
  typedef Tpetra::Map<>::global_ordinal_type     GO;
  typedef Tpetra::Map<>::node_type               Node;
  typedef Tpetra::CrsMatrix<Scalar,LO,GO>        MAT;
  typedef Tpetra::Vector<Scalar, LO, GO, Node>   V;
  typedef Tpetra::MultiVector<Scalar,LO,GO>      MV;
  typedef Tpetra::Operator<Scalar, LO, GO, Node> OP;
  typedef Tpetra::Map<LO,GO,Node>                MAP;
  typedef Belos::OperatorTraits<Scalar,MV,OP>    OPT;
  typedef Belos::MultiVecTraits<Scalar,MV>       MVT;

  Tpetra::initialize (&argc, &argv);
  auto comm = Tpetra::getDefaultComm ();

  bool verbose = false;
  bool success = true;
  try {
    bool proc_verbose = false;
    int frequency = -1;        // frequency of status test output.
    int blocksize = 1;         // blocksize, PCPGIter
    int numrhs = 1;            // number of right-hand sides to solve for
    int maxiters = 30;         // maximum number of iterations allowed per linear system

    int maxDeflate = 8; // maximum number of vectors deflated from the linear system;
    // There is no overhead cost assoc with changing maxDeflate between solves
    int maxSave = 16;    // maximum number of vectors saved from current and previous .");
    // If maxSave changes between solves, then re-initialize (setSize).

    // Hypothesis: seed vectors are conjugate.
    // Initial versions allowed users to supply a seed space et cetera, but no longer.

    // The documentation it suitable for certain tasks, like defining a modules grammar,
    std::string ortho("ICGS"); // The Belos documentation obscures the fact that
    // IMGS is Iterated Modified Gram Schmidt,
    // ICGS is Iterated Classical Gram Schmidt, and
    // DKGS is another Iterated Classical Gram Schmidt.
    // Mathematical issues, such as the difference between ICGS and DKGS, are not documented at all.
    // UH tells me that Anasazi::SVQBOrthoManager is available;  I need it for Belos
    MT tol = 1.0e-8;           // relative residual tolerance

    // How do command line parsers work?
    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters)");
    cmdp.setOption("tol",&tol,"Relative residual tolerance used by PCPG solver");
    cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size)");
    cmdp.setOption("num-deflate",&maxDeflate,"Number of vectors deflated from the linear system");
    cmdp.setOption("num-save",&maxSave,"Number of vectors saved from old Krylov subspaces");
    cmdp.setOption("ortho-type",&ortho,"Orthogonalization type, either DGKS, ICGS or IMGS");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    if (!verbose)
      frequency = -1;  // reset frequency if test is not verbose

    //
    // *************Form the problem*********************
    //
    int numElePerDirection = 14*comm->getSize(); // 5 -> 20
    int num_time_step = 4;
    int numNodes = (numElePerDirection - 1)*(numElePerDirection - 1);
    //By the way, either matrix has (3*numElePerDirection - 2)^2 nonzeros.
    RCP<MAP> map = rcp(new MAP(numNodes, 0, comm) );
    RCP<MAT> stiff = rcp(new MAT(map, 0));
    RCP<MAT> mass = rcp(new MAT(map, 0) );
    RCP<V> vecLHS = rcp( new V(map) );
    RCP<V> vecRHS = rcp( new V(map) );

    RCP<MV> LHS, RHS;
    Scalar ko = 8.0/3.0;
    Scalar k1 = -1.0/3.0;
    Scalar h =  1.0/(double) numElePerDirection;  // x=(iX,iY)h
    Scalar mo = h*h*4.0/9.0;
    Scalar m1 = h*h/9.0;
    Scalar m2 = h*h/36.0;
    double pi = 4.0 * atan(1.0), valueLHS;
    int iX, iY;
    for(LO lid = map->getMinLocalIndex(); lid <= map->getMaxLocalIndex(); lid++){
      GO node = map->getGlobalElement(lid);
      iX  = node  % (numElePerDirection-1);
      iY  = ( node - iX )/(numElePerDirection-1);
      GO pos = node;
      stiff->insertGlobalValues(node, tuple(pos), tuple(ko));
      mass->insertGlobalValues(node, tuple(pos), tuple(mo)); // init guess violates hom Dir bc
      valueLHS = sin( pi*h*((double) iX+1) )*cos( 2.0 * pi*h*((double) iY+1) );
      vecLHS->replaceGlobalValue( 1, valueLHS);
      if (iY > 0) {
        pos = iX + (iY-1)*(numElePerDirection-1);
        stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); //North
        mass->insertGlobalValues(node, tuple(pos), tuple(m1));
      }
      if (iY < numElePerDirection-2) {
        pos = iX + (iY+1)*(numElePerDirection-1);
        stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); //South
        mass->insertGlobalValues(node, tuple(pos), tuple(m1));
      }

      if (iX > 0) {
        pos = iX-1 + iY*(numElePerDirection-1);
        stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // West
        mass->insertGlobalValues(node, tuple(pos), tuple(m1));
        if (iY > 0) {
          pos = iX-1 + (iY-1)*(numElePerDirection-1);
          stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // North West
          mass->insertGlobalValues(node, tuple(pos), tuple(m2));
        }
        if (iY < numElePerDirection-2) {
          pos = iX-1 + (iY+1)*(numElePerDirection-1);
          stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // South West
          mass->insertGlobalValues(node, tuple(pos), tuple(m2));
        }
      }

      if (iX < numElePerDirection - 2) {
        pos = iX+1 + iY*(numElePerDirection-1);
        stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // East
        mass->insertGlobalValues(node, tuple(pos), tuple(m1));
        if (iY > 0) {
          pos = iX+1 + (iY-1)*(numElePerDirection-1);
          stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // North East
          mass->insertGlobalValues(node, tuple(pos), tuple(m2));
        }
        if (iY < numElePerDirection-2) {
          pos = iX+1 + (iY+1)*(numElePerDirection-1);
          stiff->insertGlobalValues(node, tuple(pos), tuple(k1)); // South East
          mass->insertGlobalValues(node, tuple(pos), tuple(m2));
        }
      }
    }
    stiff->fillComplete();
    mass->fillComplete();

    const Scalar ONE  = SCT::one();

    double hdt = .00005; // half time step
    RCP<MAT> A = rcp(new MAT(*stiff, Teuchos::Copy) );// A = Mass+Stiff*dt/2
    // int err = EpetraExt::MatrixMatrix::Add(*Mass, false, one,*A,hdt);
    
    Tpetra::MatrixMatrix::add<Scalar,LO,GO,Node>(ONE,false,*mass, Scalar(hdt), false, *A);
    A->fillComplete();

    hdt = -hdt;
    RCP<MAT> B = rcp(new MAT(*stiff, Teuchos::Copy) );// B = Mass-Stiff*dt/2
    // mass->add( ONE, B, Scalar(hdt),B->getDomainMap(), B->getRangeMap()); // Mass->add( false, one, *B, hdt);
    Tpetra::MatrixMatrix::add<Scalar,LO,GO,Node>(ONE,false,*mass, Scalar(hdt), false, *B);
    // if (err != 0) {
    //   std::cout << "err "<<err<<" from MatrixMatrix::Add "<<std::endl;
    //   return(err);
    // }
    B->fillComplete();
    // old epetra B->multiply(false, *vecLHS, *vecRHS); // rhs_new := B*lhs_old,
    B->apply(*vecLHS, *vecRHS);

    proc_verbose = verbose && (comm->getRank()==0);  /* Only print on the zero processor */

    LHS = Teuchos::rcp_implicit_cast<MV>(vecLHS);
    RHS = Teuchos::rcp_implicit_cast<MV>(vecRHS);

    //
    // ************Construct preconditioner*************
    //
    // Teuchos::ParameterList MLList; // Set MLList for Smoothed Aggregation

    // ML_Tpetra::SetDefaults("SA", MLList); // reset parameters ML User's Guide
    // MLList.set("smoother: type","Chebyshev"); // Chebyshev smoother  ... aztec??
    // MLList.set("smoother: sweeps",3);
    // MLList.set("smoother: pre or post", "both"); // both pre- and post-smoothing

// TD: For Tpetra What this code is doing (Tpetra alternative to Amesos/KLU seems to be Amesos2/KLU2)
// 
// #ifdef HAVE_ML_AMESOS
//     MLList.set("coarse: type","Amesos-KLU"); // solve with serial direct solver KLU
// #else
//     MLList.set("coarse: type","Jacobi");     // not recommended
//     puts("Warning: Iterative coarse grid solve");
// #endif

    //
    //ML_Epetra::MultiLevelPreconditioner* Prec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
    // RCP<OP> prec = MueLu::CreateTpetraPreconditioner(A, MLList );
    // assert(prec != Teuchos::null);

    // Create the Belos preconditioned operator from the preconditioner.
    // NOTE:  This is necessary because Belos expects an operator to apply the
    //        preconditioner with Apply() NOT ApplyInverse().
    
    // RCP<Belos::EpetraPrecOp> belosPrec = rcp( new Belos::EpetraPrecOp( Prec ) );
    // RCP<Belos::TpetraOperator> belosPrec = rcp( new Belos::TpetraOperator( prec ) );

    //
    // *****Create parameter list for the PCPG solver manager*****
    //
    const int numGlobalElements = RHS->getGlobalLength();
    if (maxiters == -1)
      maxiters = numGlobalElements/blocksize - 1; // maximum number of iterations to run
    //
    Teuchos::ParameterList belosList;
    belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    belosList.set( "Num Deflated Blocks", maxDeflate );    // Number of vectors in seed space
    belosList.set( "Num Saved Blocks", maxSave );          // Number of vectors saved from old spaces
    belosList.set( "Orthogonalization", ortho );           // Orthogonalization type

    if (numrhs > 1) {
      belosList.set( "Show Maximum Residual Norm Only", true );  // although numrhs = 1.
    }
    if (verbose) {
      belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
          Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails );
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    else
      belosList.set( "Verbosity", Belos::Errors + Belos::Warnings + Belos::FinalSummary );
    //
    // *******Construct a preconditioned linear problem********
    //
    RCP<Belos::LinearProblem<double,MV,OP> > problem
      = rcp( new Belos::LinearProblem<double,MV,OP>( A, LHS, RHS ) );
    
    //problem->setLeftPrec( belosPrec );

    bool set = problem->setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return -1;
    }

    // Create an iterative solver manager.
    RCP< Belos::SolverManager<double,MV,OP> > solver
      = rcp( new Belos::PCPGSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );

    // std::cout <<  LHS.Values() << std::endl

    //
    // *******************************************************************
    // ************************* Iterate PCPG ****************************
    // *******************************************************************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << numGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Block size used by solver: " << blocksize << std::endl;
      std::cout << "Maximum number of iterations allowed: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    bool badRes;
    for( int time_step = 0; time_step < num_time_step; time_step++){
      if( time_step ){
        // old epetra B->multiply(false, *vecLHS, *vecRHS); // rhs_new := B*lhs_old,
        B->apply(*LHS, *RHS); // rhs_new := B*lhs_old,
        set = problem->setProblem(LHS,RHS);
        if (set == false) {
          if (proc_verbose)
            std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
          return -1;
        }
      } // if time_step
      std::vector<double> rhs_norm( numrhs );
      MVT::MvNorm( *RHS, rhs_norm );
      std::cout << "                  RHS norm is ... " << rhs_norm[0] << std::endl;
      //
      // Perform solve
      //
      Belos::ReturnType ret = solver->solve();
      //
      // Compute actual residuals.
      //
      badRes = false;
      std::vector<double> actual_resids( numrhs );
      //std::vector<double> rhs_norm( numrhs );
      MV resid(map, numrhs); // Epetra_MultiVector resid(*Map, numrhs);
      OPT::Apply( *A, *LHS, resid );
      MVT::MvAddMv( -1.0, resid, 1.0, *RHS, resid );
      MVT::MvNorm( resid, actual_resids );
      MVT::MvNorm( *RHS, rhs_norm );
      std::cout << "                    RHS norm is ... " << rhs_norm[0] << std::endl;

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
        break;
      }
    } // for time_step
    if (proc_verbose) {
      if (success)
        std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
      else
        std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

#ifdef EPETRA_MPI
  MPI_Finalize();
#endif

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
