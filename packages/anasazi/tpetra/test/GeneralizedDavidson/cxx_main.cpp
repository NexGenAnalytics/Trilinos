// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
//                 Copyright 2004 Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// ***********************************************************************
// @HEADER
//
// This test is for GeneralizedDavidson solving a generalized (Ax=Bxl) real Hermitian
// eigenvalue problem, using the GeneralizedDavidsonSolMgr solver manager.
// A few steps of BlockPCG is used as a preconditioner.
//

// This code does not compile due to the use of ModeLaplace1DQ1,
// which depends on Epetra. It is commented out in CMakeLists.

// Tpetra
#include <Tpetra_Map.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_CrsMatrix.hpp>

// Teuchos
#include "Teuchos_CommandLineProcessor.hpp"

// Anasazi
#include "AnasaziTypes.hpp"
#include "AnasaziBasicSort.hpp"
#include "AnasaziConfigDefs.hpp"
#include "AnasaziTpetraAdapter.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziGeneralizedDavidsonSolMgr.hpp"

#include "ModeLaplace1DQ1.h"
#include "BlockPCGSolver.h"

using namespace Teuchos;

template <typename ScalarType>
int run (int argc, char *argv[]) {

  using ST  = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO  = typename Tpetra::MultiVector<>::local_ordinal_type;
  using GO  = typename Tpetra::MultiVector<>::global_ordinal_type;
  using NT  = typename Tpetra::MultiVector<>::node_type;

  using MT = typename Teuchos::ScalarTraits<ST>::MT;

  using OP  = Tpetra::Operator<ST,LO,GO,NT>;
  using MV  = Tpetra::MultiVector<ST,LO,GO,NT>;
  using OPT = Anasazi::OperatorTraits<ST,MV,OP>;
  using MVT = Anasazi::MultiVecTraits<ST,MV>;
  using SCT = Teuchos::ScalarTraits<ST>;

  using tmap_t = Tpetra::Map<LO,GO,NT>;
  using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;

  bool boolret;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
  const auto comm = Tpetra::getDefaultComm();

  int MyPID = comm->getRank();

  bool testFailed;
  bool verbose = false;
  bool debug = false;
  std::string which("LM");

  CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("debug","nodebug",&debug,"Print debugging information.");
  cmdp.setOption("sort",&which,"Targetted eigenvalues (SM or LM).");
  if (cmdp.parse(argc,argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {

    return -1;
  }
  if (debug) verbose = true;

  const ST ONE  = SCT::one();

  if (verbose && MyPID == 0) {
    std::cout << Anasazi::Anasazi_Version() << std::endl << std::endl;
  }

  //  Problem information
  int space_dim = 1;
  std::vector<double> brick_dim( space_dim );
  brick_dim[0] = 1.0;
  std::vector<int> elements( space_dim );
  elements[0] = 100;

  // Create problem
  RCP<ModalProblem> testCase = rcp( new ModeLaplace1DQ1(comm, brick_dim[0], elements[0]) );
  //
  // Get the stiffness and mass matrices
  RCP<tcrsmatrix_t> K = rcp( const_cast<tcrsmatrix_t *>(testCase->getStiffness()), false );
  RCP<tcrsmatrix_t> M = rcp( const_cast<tcrsmatrix_t *>(testCase->getMass()), false );
  //
  // Create solver for mass matrix
  // Note that accuracy of Davidson solution does NOT depend on how accurately the BlockPCG is solved
  const int maxIterCG = 10;
  const double tolCG = 1e-2;

  RCP<BlockPCGSolver> opStiffness = rcp( new BlockPCGSolver(comm, M.get(), tolCG, maxIterCG, 0) );
  opStiffness->setPreconditioner( 0 );
  RCP<OP> invStiffness = rcp( new OP(opStiffness.get()) );

  // Create the initial vectors
  int blockSize = 3;
  RCP<MV> ivec = rcp( new MV(K->OperatorDomainMap(), blockSize) );
  MVT::MvRandom( *ivec );

  // Create eigenproblem
  const int nev = 5;
  RCP<Anasazi::BasicEigenproblem<ST,MV,OP> > problem =
    rcp( new Anasazi::BasicEigenproblem<ST,MV,OP>() );
  problem->setA(K);
  problem->setM(M);
  problem->setPrec(invStiffness);
  problem->setInitVec(ivec);
  //
  // Set the number of eigenvalues requested
  problem->setNEV( nev );
  //
  // Inform the eigenproblem that you are done passing it information
  boolret = problem->setProblem();
  if (boolret != true) {
    if (verbose && MyPID == 0) {
      std::cout << "Anasazi::BasicEigenproblem::SetProblem() returned with error." << std::endl
           << "End Result: TEST FAILED" << std::endl;
    }
    return -1;
  }

  // Set verbosity level
  int verbosity = Anasazi::Errors + Anasazi::Warnings;
  if (verbose) {
    verbosity += Anasazi::FinalSummary + Anasazi::TimingDetails;
  }
  if (debug) {
    verbosity += Anasazi::Debug;
  }


  // Eigensolver parameters
  int maxRestarts = 25;
  int maxDim = 50;
  MT tol = 1e-6;
  //
  // Create parameter list to pass into the solver manager
  ParameterList MyPL;
  MyPL.set( "Verbosity", verbosity );
  MyPL.set( "Which", which );
  MyPL.set( "Maximum Subspace Dimension", maxDim );
  MyPL.set( "Block Size", blockSize );
  MyPL.set( "Maximum Restarts", maxRestarts );
  MyPL.set( "Convergence Tolerance", tol );
  //
  // Create the solver manager
  Anasazi::GeneralizedDavidsonSolMgr<ST,MV,OP> MySolverMgr(problem, MyPL);
  //
  // Check that the parameters were all consumed
  if (MyPL.getEntryPtr("Verbosity")->isUsed() == false ||
      MyPL.getEntryPtr("Which")->isUsed() == false ||
      MyPL.getEntryPtr("Maximum Subspace Dimension")->isUsed() == false ||
      MyPL.getEntryPtr("Block Size")->isUsed() == false ||
      MyPL.getEntryPtr("Maximum Restarts")->isUsed() == false ||
      MyPL.getEntryPtr("Convergence Tolerance")->isUsed() == false) {
    if (verbose && MyPID==0) {
      std::cout << "Failure! Unused parameters: " << std::endl;
      MyPL.unused(std::cout);
    }
  }


  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode = MySolverMgr.solve();
  testFailed = false;
  if (returnCode != Anasazi::Converged) {
    testFailed = true;
  }

  // Get the eigenvalues and eigenvectors from the eigenproblem
  Anasazi::Eigensolution<ST,MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<ST> > evals = sol.Evals;
  RCP<MV> evecs = sol.Evecs;
  int numev = sol.numVecs;

  if (numev > 0) {

    std::ostringstream os;
    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(6);

    // Compute the direct residual
    std::vector<ST> normV( numev );
    SerialDenseMatrix<int,ST> T(numev,numev);
    for (int i=0; i<numev; i++) {
      T(i,i) = evals[i].realpart;
    }
    RCP<MV> Mvecs = MVT::Clone( *evecs, numev );
    RCP<MV> Kvecs = MVT::Clone( *evecs, numev );
    OPT::Apply( *K, *evecs, *Kvecs );
    OPT::Apply( *M, *evecs, *Mvecs );
    MVT::MvTimesMatAddMv( -ONE, *Mvecs, T, ONE, *Kvecs );
    // compute 2-norm of residuals
    std::vector<MT> resnorm(numev);
    MVT::MvNorm( *Kvecs, resnorm );

    os << "Number of iterations performed in GeneralizedDavidson_test.exe: " << MySolverMgr.getNumIters() << std::endl
       << "Direct residual norms computed in GeneralizedDavidson_test.exe" << std::endl
       << std::setw(20) << "Eigenvalue" << std::setw(20) << "Residual" << std::endl
       << "----------------------------------------" << std::endl;
    for (int i=0; i<numev; i++) {
      os << std::setw(20) << evals[i].realpart << std::setw(20) << resnorm[i] << std::endl;
      if ( resnorm[i] > tol ) {
        testFailed = true;
      }
    }
    if (verbose && MyPID==0) {
      std::cout << std::endl << os.str() << std::endl;
    }
  }

  if (testFailed) {
    if (verbose && MyPID==0) {
      std::cout << "End Result: TEST FAILED" << std::endl;
    }
    return -1;
  }
  //
  // Default return value
  //
  if (verbose && MyPID==0) {
    std::cout << "End Result: TEST PASSED" << std::endl;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  return run<double>(argc,argv);
  // run<float>(argc,argv);
}
