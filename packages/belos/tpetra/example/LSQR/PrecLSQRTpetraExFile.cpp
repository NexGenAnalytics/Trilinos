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
//
// This driver reads a problem from a file, which can be in Harwell-Boeing (*.hb),
// Matrix Market (*.mtx), or triplet format (*.triU, *.triS).  The right-hand side
// from the problem, if it exists, will be used instead of multiple random
// right-hand-sides.  The initial guesses are all set to zero.  An ILU preconditioner
// is constructed using the Ifpack factory.
//
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosLSQRSolMgr.hpp"
#include "BelosTpetraTestFramework.hpp"

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include "Ifpack2_Factory.hpp"
#include "Ifpack2_Preconditioner.hpp"
#include "Ifpack2_Parameters.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

void IFPACK2_CHK_ERR(int code) {
  if (code < 0) { \
  std::cerr << "IFPACK2 ERROR " << code << ", " \
    << __FILE__ << ", line " << __LINE__ << std::endl; \
    return(code);  }
}

template<typename ScalarType>
int run(int argc, char *argv[]) {
  using Teuchos::CommandLineProcessor;
  using Teuchos::GlobalMPISession;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_implicit_cast;
  using Teuchos::tuple;

  using ST  = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO  = typename Tpetra::Vector<>::local_ordinal_type;
  using GO  = typename Tpetra::Vector<>::global_ordinal_type;
  using NT  = typename Tpetra::Vector<>::node_type;

  using V   = typename Tpetra::Vector<ST,LO,GO,NT>;
  using MV  = typename Tpetra::MultiVector<ST,LO,GO,NT>;
  using OP  = typename Tpetra::Operator<ST,LO,GO,NT>;
  using MAP = typename Tpetra::Map<LO,GO,NT>;
  using MAT = typename Tpetra::CrsMatrix<ST,LO,GO,NT>;

  using MVT = typename Belos::MultiVecTraits<ST,MV>;
  using OPT = typename Belos::OperatorTraits<ST,MV,OP>;

  using MT  = typename Teuchos::ScalarTraits<ST>::magnitudeType;

  using Ifpack2Prec = typename Ifpack2::Preconditioner<ST,LO,GO,NT>;

  Teuchos::GlobalMPISession session(&argc, &argv, NULL);
  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();

  bool verbose = false;
  bool success = true;

  try {
    bool proc_verbose = false;
    bool leftprec = true;      // left preconditioning or right.
    // LSQR applies the operator and the transposed operator.
    // A preconditioner must support transpose multiply.
    int frequency = -1;        // frequency of status test output.
    int blocksize = 1;         // blocksize
    // LSQR as currently implemented is a single vector algorithm.
    // However some of the parameters that would be used by a block version
    // have not been removed from this file.
    int numRHS = 1;            // number of right-hand sides to solve for
    int maxiters = -1;         // maximum number of iterations allowed per linear system
    std::string filename("orsirr1_scaled.hb");
    MT relResTol = 1.0e-5;     // relative residual tolerance for the preconditioned linear system
    MT resGrowthFactor = 1.0;  // In this example, warn if |resid| > resGrowthFactor * relResTol

    MT relMatTol = 1.e-10;     // relative Matrix error, default value sqrt(eps)
    MT maxCond  = 1.e+5;       // maximum condition number default value 1/eps
    MT damp = 0.;              // regularization (or damping) parameter

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("left-prec","right-prec",&leftprec,"Left preconditioning or right.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
    cmdp.setOption("lambda",&damp,"Regularization parameter");
    cmdp.setOption("tol",&relResTol,"Relative residual tolerance");
    cmdp.setOption("matrixTol",&relMatTol,"Relative error in Matrix");
    cmdp.setOption("max-cond",&maxCond,"Maximum condition number");
    cmdp.setOption("num-rhs",&numRHS,"Number of right-hand sides to be solved for.");
    cmdp.setOption("block-size",&blocksize,"Block size used by LSQR.");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    if (!verbose)
      frequency = -1;  // reset frequency if test is not verbose

    //
    // *************Get the problem*********************
    //
    Belos::Tpetra::HarwellBoeingReader<MAT> reader( comm );
    RCP<MAT> A = reader.readFromFile( filename );
    RCP<const MAP> map = A->getDomainMap();

    // Initialize vectors
    RCP<MV> vecB = rcp(new MV(map, numRHS));
    RCP<MV> vecX = rcp(new MV(map, numRHS));
    RCP<MV> B, X;

    proc_verbose = verbose && (comm->getRank()==0);  /* Only print on the zero processor */

    // Check to see if the number of right-hand sides is the same as requested.
    if (numRHS>1) {
      X = rcp( new MV( map, numRHS ) );
      B = rcp( new MV( map, numRHS ) );
      X->randomize();
      OPT::Apply( *A, *X, *B );
      X->putScalar( 0.0 );
    }
    else {
      int locNumCol = map->getMaxLocalIndex() + 1; // Create a known solution
      int globNumCol = map->getMaxGlobalIndex() + 1;
      for( int li = 0; li < locNumCol; li++){   // assume consecutive lid
        const auto gid = map->getGlobalElement(li);
        ST value = (ST) ( globNumCol -1 - gid );
        int numEntries = 1;
        vecX->replaceGlobalValue(numEntries,0,value);
      }
      A->apply( *vecX, *vecB ); // Create a consistent linear system
  
      // At this point, the initial guess is exact.
      bool zeroInitGuess = false; // annihilate initial guess
      bool goodInitGuess = true; // initial guess near solution
      if ( zeroInitGuess ) {
        vecX->putScalar( 0.0 );
      } else {
          if( goodInitGuess )
            {
              ST value = 1.e-2; // "Rel RHS Err" and "Rel Mat Err" apply to the residual equation,
              // LO numEntries = 1;   // norm( b - A x_k ) ?<? relResTol norm( b- Axo).
              LO index = 0;        // norm(b) is inaccessible to LSQR.
              vecX->sumIntoLocalValue(index, 0, value);
            }
        }
      X = vecX;
      B = vecB;
    }
    //
    // ************Construct preconditioner*************
    //
    ParameterList ifpack2List;

    // create the preconditioner. For valid PrecType values,
    // please check the documentation
    std::string PrecType = "ILU"; // incomplete LU
    int OverlapLevel = 1; // nonnegative

    RCP<Ifpack2Prec> Prec = Ifpack2::Factory::create<MAT>("ILUT", Teuchos::rcpFromRef(A), OverlapLevel);
    assert(Prec != Teuchos::null);

    // specify parameters for ILU
    ifpack2List.set("fact: level-of-fill", 1);
    // the combine mode is on the following:
    // "Add", "Zero", "Insert", "InsertAdd", "Average", "AbsMax"
    // Their meaning is as defined in file Epetra_CombineMode.h
    ifpack2List.set("schwarz: combine mode", "Add");
    // sets the parameters
    IFPACK2_CHK_ERR(Prec->setParameters(ifpack2List));

    // initialize the preconditioner. At this point the matrix must
    // have been FillComplete()'d, but actual values are ignored.
    IFPACK2_CHK_ERR(Prec->initialize());

    // Builds the preconditioners, by looking for the values of
    // the matrix.
    IFPACK2_CHK_ERR(Prec->compute());

    {
      const int errcode = Prec->SetUseTranspose (true);
      if (errcode != 0) {
        throw std::logic_error ("Oh hai! Ifpack_Preconditioner doesn't know how to apply its transpose.");
      } else {
        (void) Prec->SetUseTranspose (false);
      }
    }

    // Create the Belos preconditioned operator from the Ifpack preconditioner.
    // NOTE:  This is necessary because Belos expects an operator to apply the
    //        preconditioner with Apply() NOT ApplyInverse().
    // TODO: Find alternative of EpetraPrecOp for Tpetra ?
    // RCP<Belos::EpetraPrecOp> belosPrec = rcp( new Belos::EpetraPrecOp( Prec ) );
    RCP<Ifpack2Prec> belosPrec = Prec;

    //
    // *****Create parameter list for the LSQR solver manager*****
    //
    const int numGlobalElements = B->getGlobalNumElements();
    if (maxiters == -1)
      maxiters = numGlobalElements/blocksize - 1; // maximum number of iterations to run
    //
    ParameterList belosList;
    belosList.set( "Block Size", blocksize );       // Blocksize to be used by iterative solver
    belosList.set( "Lambda", damp );                // Regularization parameter
    belosList.set( "Rel RHS Err", relResTol );      // Relative convergence tolerance requested
    belosList.set( "Rel Mat Err", relMatTol );      // Maximum number of restarts allowed
    belosList.set( "Condition Limit", maxCond);     // upper bound for cond(A)
    belosList.set( "Maximum Iterations", maxiters );// Maximum number of iterations allowed
    if (numRHS > 1) {
      belosList.set( "Show Maximum Residual Norm Only", true );  // Show only the maximum residual norm
    }
    if (verbose) {
      belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
        Belos::TimingDetails + Belos::StatusTestDetails );
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    else
      belosList.set( "Verbosity", Belos::Errors + Belos::Warnings );
    //
    // *******Construct a preconditioned linear problem********
    //
    RCP<Belos::LinearProblem<double,MV,OP> > problem
      = rcp( new Belos::LinearProblem<double,MV,OP>( A, X, B ) );
    if (leftprec) {
      problem->setLeftPrec( belosPrec );
    }
    else {
      problem->setRightPrec( belosPrec );
    }
    bool set = problem->setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return -1;
    }

    // Create an iterative solver manager.
    RCP< Belos::LSQRSolMgr<double,MV,OP> > solver
      = rcp( new Belos::LSQRSolMgr<double,MV,OP>(problem, rcp(&belosList,false)));

    //
    // *******************************************************************
    // ******************Start the LSQR iteration*************************
    // *******************************************************************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << numGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numRHS << std::endl;
      std::cout << "Block size used by solver: " << blocksize << std::endl;
      std::cout << "Max number of Gmres iterations per restart cycle: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << relResTol << std::endl;
      std::cout << std::endl;
      std::cout << "Solver's Description: " << std::endl;
      std::cout << solver->description() << std::endl; // visually verify the parameter list
    }
    //
    // Perform solve
    //
    Belos::ReturnType ret = solver->solve();
    //
    // Get the number of iterations for this solve.
    //
    std::vector<double> solNorm( numRHS );      // get solution norm
    MVT::MvNorm( *X, solNorm );
    int numIters = solver->getNumIters();
    MT condNum = solver->getMatCondNum();
    MT matrixNorm= solver->getMatNorm();
    MT resNorm = solver->getResNorm();
    MT lsResNorm = solver->getMatResNorm();
    if (proc_verbose)
      std::cout << "Number of iterations performed for this solve: " << numIters << std::endl
      << "matrix condition number: " << condNum << std::endl
      << "matrix norm: " << matrixNorm << std::endl
      << "residual norm: " << resNorm << std::endl
      << "solution norm: " << solNorm[0] << std::endl
      << "least squares residual Norm: " << lsResNorm << std::endl;
    //
    // Compute actual residuals.
    //
    bool badRes = false;
    std::vector<double> actual_resids( numRHS );
    std::vector<double> rhs_norm( numRHS );
    MV resid(map, numRHS);
    OPT::Apply( *A, *X, resid );
    MVT::MvAddMv( -1.0, resid, 1.0, *B, resid );
    MVT::MvNorm( resid, actual_resids );
    MVT::MvNorm( *B, rhs_norm );
    if (proc_verbose) {
      std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
      for ( int i=0; i<numRHS; i++) {
        double actRes = actual_resids[i]/rhs_norm[i];
        std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
        if (actRes > relResTol * resGrowthFactor ) badRes = true;
      }
    }

    if (ret!=Belos::Converged || badRes) {
      success = false;
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
    } else {
      success = true;
      if (proc_verbose)
        std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
  run<double>(argc,argv);
} // end PrecLSQRTpetraExFile.cpp
