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
//  This test is for the generalized Davidsoneigensolver
//
//  This code does not compile due to the use of ModeLaplace1DQ1,
//  which depends on Epetra. It is commented out in CMakeLists.

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>

// Teuchos
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_CommandLineProcessor.hpp"

// Anasazi

#include "AnasaziTypes.hpp"
#include "AnasaziBasicSort.hpp"
#include "AnasaziConfigDefs.hpp"
#include "AnasaziSolverUtils.hpp"
#include "AnasaziTpetraAdapter.hpp"
#include "AnasaziSVQBOrthoManager.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "AnasaziStatusTestMaxIters.hpp"
#include "AnasaziGeneralizedDavidson.hpp"

#include "ModeLaplace1DQ1.h"


using namespace Teuchos;
using namespace Anasazi;


class get_out : public std::logic_error {
  public: get_out(const std::string &whatarg) : std::logic_error(whatarg) {}
};

template <typename ST, typename MT, class MV, class MVT, class OP>
void checks( RCP<GeneralizedDavidson<ST,MV,OP> > solver, int blockSize, int maxdim,
             RCP<Eigenproblem<ST,MV,OP> > problem,
             RCP<MatOrthoManager<ST,MV,OP> > ortho) {
  GeneralizedDavidsonState<ST,MV> state = solver->getState();

  TEUCHOS_TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*state.V) != solver->getMaxSubspaceDim(),get_out,"getMaxSubspaceDim() does not match allocated size for V");

  TEUCHOS_TEST_FOR_EXCEPTION(&solver->getProblem() != problem.get(),get_out,"getProblem() did not return the submitted problem.");

  if (solver->isInitialized())
  {
      // Generalized Davidson block size is equal to or one greater than user specified block size
      // Because GeneralizedDavidson block size is variable, this check only applied to an initialized solver
      TEUCHOS_TEST_FOR_EXCEPTION(solver->getBlockSize() != blockSize && solver->getBlockSize() != blockSize+1, get_out,"Solver block size does not match specified block size.");

    std::vector<Anasazi::Value<ST> > ritzValues = solver->getRitzValues();

    // check Ritz residuals
    std::vector<MT> ritzResids = solver->getRitzRes2Norms();

    // get Ritz index
    std::vector<int> ritzIndex = solver->getRitzIndex();

    // get Ritz vector
    RCP<const MV> ritzVectors = solver->getRitzVectors();

    int numRitzVecs = MVT::GetNumberVecs(*ritzVectors);

    RCP<MV> tmpVecs = MVT::Clone( *ritzVectors, numRitzVecs );

    // Compute Ritz residuals like R = A*X - B*X*T
    Teuchos::SerialDenseMatrix<int,ST> T(numRitzVecs,numRitzVecs);
    Teuchos::RCP<MV> ritzResiduals = MVT::Clone( *ritzVectors, numRitzVecs );
    for (int i=0; i<T.numRows(); i++) T(i,i) = ritzValues[i].realpart;
    OP::Apply( *(problem->getA()), *ritzVectors, *ritzResiduals );
    if( problem->getM() != Teuchos::null )
    {
        OP::Apply( *(problem->getM()), *ritzVectors, *tmpVecs );
    }
    else
    {
        std::vector<int> inds(numRitzVecs);
        for( int i=0; i<numRitzVecs; ++i ) inds[i]=i;
        MVT::SetBlock( *ritzVectors, inds, *tmpVecs );
    }
    MVT::MvTimesMatAddMv(-1.0,*tmpVecs,T,1.0,*ritzResiduals);

    // Compute the norm of the Ritz residual vectors
    std::vector<MT> ritzVecNrm( numRitzVecs );
    MVT::MvNorm( *ritzVectors, ritzVecNrm );
    MT error;
    for (int i=0; i<numRitzVecs; i++) {
      error = Teuchos::ScalarTraits<MT>::magnitude( ritzVecNrm[i] - 1.0 );
      TEUCHOS_TEST_FOR_EXCEPTION(error > 1e-14,get_out,"Ritz vectors are not normalized.");
    }

    std::vector<MT> ritzResNrm( MVT::GetNumberVecs( *ritzResiduals ) );
    MVT::MvNorm( *ritzResiduals, ritzResNrm );
    for (int i=0; i<(int)ritzResNrm.size(); i++) {
      error = Teuchos::ScalarTraits<MT>::magnitude( ritzResids[i] - ritzResNrm[i] );
      TEUCHOS_TEST_FOR_EXCEPTION(error > 1e-12,get_out,"Ritz residuals from iteration do not compare to those computed.");
    }
  }
  else {
    // not initialized
    TEUCHOS_TEST_FOR_EXCEPTION(solver->getCurSubspaceDim() != 0,get_out,"In unitialized state, getCurSubspaceDim() should be 0.");
  }
}

template <typename ST, typename MT, class MVT, class MV, class OP>
void testsolver( RCP<BasicEigenproblem<ST,MV,OP> > problem,
                 RCP< OutputManager<ST> > printer,
                 RCP< MatOrthoManager<ST,MV,OP> > ortho,
                 RCP< SortManager<MT> > sorter,
                 ParameterList &pls,bool invalid,
                 GeneralizedDavidsonState<ST,MV> initstate, bool invalidinit)
{
  // create a status tester
  RCP< StatusTest<ST,MV,OP> > tester = rcp( new StatusTestMaxIters<ST,MV,OP>(1) );

  // create the solver
  RCP< GeneralizedDavidson<ST,MV,OP> > solver;
  try {
    solver = rcp( new GeneralizedDavidson<ST,MV,OP>(problem,sorter,printer,tester,ortho,pls) );
    TEUCHOS_TEST_FOR_EXCEPTION(invalid, get_out, "Instantiating with invalid parameters failed to throw exception.")
  }
  catch (const std::invalid_argument &ia) {
    if (!invalid) {
      printer->stream(Warnings) << "Error thrown at instantiation: " << ia.what() << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!invalid, get_out, "Instantiating with valid parameters unexpectadly threw exception.");

    // caught expected exception
    return;
  }

  const int  blockSize = pls.get<int>("Block Size");
  const int  maxdim = pls.get<int>("Maximum Subspace Dimension");

  SolverUtils<ST,MV,OP> msutils;

  // solver should be uninitialized
  TEUCHOS_TEST_FOR_EXCEPTION(solver->isInitialized() != false,get_out,"Solver should be un-initialized after instantiation.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getNumIters() != 0,get_out,"Number of iterations after initialization should be zero after init.")
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getAuxVecs().size() != 0,get_out,"getAuxVecs() should return empty.");
  checks<ST,MT,MV,MVT,OP>(solver,blockSize,maxdim,problem,ortho);

  // initialize solver and perform checks
  try {
    solver->initialize(initstate);
    TEUCHOS_TEST_FOR_EXCEPTION(invalidinit, get_out, "Initializing with invalid data failed to throw exception.")
  }
  catch (const std::invalid_argument &ia) {
    TEUCHOS_TEST_FOR_EXCEPTION(!invalidinit, get_out, "Initializing with valid data unexpectadly threw exception.");
    // caught expected exception
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(solver->isInitialized() != true,get_out,"Solver should be initialized after call to initialize().");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getNumIters() != 0,get_out,"Number of iterations should be zero.")
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getAuxVecs().size() != 0,get_out,"getAuxVecs() should return empty.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getCurSubspaceDim() != blockSize,get_out,"after init, getCurSubspaceDim() should be equal to block size.");
  checks<ST,MT,MV,MVT,OP>(solver,blockSize,maxdim,problem,ortho);

  // call iterate(); solver should perform exactly one iteration and return; status test should be passed
  solver->iterate();
  TEUCHOS_TEST_FOR_EXCEPTION(tester->getStatus() != Passed,get_out,"Solver returned from iterate() but getStatus() not Passed.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->isInitialized() != true,get_out,"Solver should be initialized after call to initialize().");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getNumIters() != 1,get_out,"Number of iterations should be one.")
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getAuxVecs().size() != 0,get_out,"getAuxVecs() should return empty.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getCurSubspaceDim() != 2*blockSize,get_out,"after one step, getCurSubspaceDim() should be 2*blockSize.");
  checks<ST,MT,MV,MVT,OP>(solver,blockSize,maxdim,problem,ortho);

  // reset numiters, call iterate(); solver should perform exactly one iteration and return; status test should be passed
  solver->resetNumIters();
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getNumIters() != 0,get_out,"Number of iterations should be zero after resetNumIters().")
  solver->iterate();
  TEUCHOS_TEST_FOR_EXCEPTION(tester->getStatus() != Passed,get_out,"Solver returned from iterate() but getStatus() not Passed.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->isInitialized() != true,get_out,"Solver should be initialized after call to initialize().");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getNumIters() != 0,get_out,"Number of iterations should be zero.")
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getAuxVecs().size() != 0,get_out,"getAuxVecs() should return empty.");
  TEUCHOS_TEST_FOR_EXCEPTION(solver->getCurSubspaceDim() != 2*blockSize,get_out,"after two steps, getCurSubspaceDim() should be 2*blockSize.");
  checks<ST,MT,MV,MVT,OP>(solver,blockSize,maxdim,problem,ortho);
}

template <typename ScalarType>
int run(int argc, char *argv[])
{
  using ST  = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO  = typename Tpetra::MultiVector<>::local_ordinal_type;
  using GO  = typename Tpetra::MultiVector<>::global_ordinal_type;
  using NT  = typename Tpetra::MultiVector<>::node_type;

  using MT = typename Teuchos::ScalarTraits<ScalarType>::magnitudeType;

  using OP  = Tpetra::Operator<ST,LO,GO,NT>;
  using MV  = Tpetra::MultiVector<ST,LO,GO,NT>;
  using OPT = Anasazi::OperatorTraits<ST,MV,OP>;
  using MVT = Anasazi::MultiVecTraits<ST,MV>;

  using tmap_t = Tpetra::Map<LO,GO,NT>;
  using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
  const auto comm = Tpetra::getDefaultComm();

  bool testFailed;
  bool verbose = false;
  bool debug = false;

  CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("debug","nodebug",&debug,"Print debugging output from iteration.");
  if (cmdp.parse(argc,argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  if (debug) verbose = true;

  // create the output manager
  int verbosity = Anasazi::Errors;
  if (verbose) {
    verbosity += Anasazi::Warnings;
  }
  if (debug) {
    verbosity += Anasazi::Debug;
  }

  RCP< OutputManager<ST> > printer =
    rcp( new BasicOutputManager<ST>( verbosity ) );

  printer->stream(Debug) << Anasazi_Version() << std::endl;

  //  Problem information
  int space_dim = 1;
  std::vector<ST> brick_dim( space_dim );
  brick_dim[0] = 1.0;
  std::vector<int> elements( space_dim );
  elements[0] = 100+1;

  // Create problem
  RCP<ModalProblem> testCase = rcp( new ModeLaplace1DQ1(comm, brick_dim[0], elements[0]) );
  //
  // Get the stiffness and mass matrices
  RCP<const tcrsmatrix_t> K = rcp( const_cast<tcrsmatrix_t *>(testCase->getStiffness()), false );
  RCP<const tcrsmatrix_t> M = rcp( const_cast<tcrsmatrix_t *>(testCase->getMass()), false );

  //
  // Create the initial vectors
  const int nev = 5;
  RCP<MV> ivec = rcp( new MV(K->OperatorDomainMap(), nev) );
  MVT::MvRandom( *ivec );
  //
  // Create eigenproblem: one standard and one generalized
  RCP<BasicEigenproblem<ST,MV,OP> > probstd = rcp( new BasicEigenproblem<ST, MV, OP>() );
  probstd->setA(K);
  probstd->setInitVec(ivec);
  RCP<BasicEigenproblem<ST,MV,OP> > probgen = rcp( new BasicEigenproblem<ST, MV, OP>() );
  probgen->setA(K);
  probgen->setM(M);
  probgen->setInitVec(ivec);
  //
  // Inform the eigenproblem that the operator A is not symmetric (even though it is)
  probstd->setHermitian(false);
  probgen->setHermitian(false);
  //
  // Set the number of eigenvalues requested
  probstd->setNEV( nev );
  probgen->setNEV( nev );
  //
  // Inform the eigenproblem that you are finishing passing it information
  if ( probstd->setProblem() != true || probgen->setProblem() != true ) {
    printer->stream(Warnings) << "Anasazi::BasicEigenproblem::SetProblem() returned with error." << std::endl
      << "End Result: TEST FAILED" << std::endl;
    return -1;
  }

  // create the orthogonalization managers: one standard and one M-based
  RCP< MatOrthoManager<ST,MV,OP> > orthostd = rcp( new SVQBOrthoManager<ST,MV,OP>() );
  RCP< MatOrthoManager<ST,MV,OP> > orthogen = rcp( new SVQBOrthoManager<ST,MV,OP>() );
  // create the sort manager
  RCP< SortManager<MT> > sorter = rcp( new BasicSort<MT>("LM") );
  // create the parameter list specifying blockSize > nev and full orthogonalization
  ParameterList pls;

  // begin testing
  testFailed = false;

  try
  {
    GeneralizedDavidsonState<ST,MV> istate;

    pls.set<int>("Block Size",nev);
    pls.set<int>("Maximum Subspace Dimension",3*nev);
    pls.set<int>("Number of Ritz Vectors",nev);
    printer->stream(Warnings) << "Testing solver(nev,3*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,false,istate,false);
    pls.set<int>("Maximum Subspace Dimension",3*nev);
    printer->stream(Warnings) << "Testing solver(nev,3*nev) with generalized eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probgen,printer,orthogen,sorter,pls,false,istate,false);

    pls.set<int>("Block Size",nev);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(nev,4*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,false,istate,false);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(nev,4*nev) with generalized eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probgen,printer,orthogen,sorter,pls,false,istate,false);

    pls.set<int>("Block Size",2*nev);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(2*nev,4*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,false,istate,false);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(2*nev,4*nev) with generalized eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probgen,printer,orthogen,sorter,pls,false,istate,false);

    pls.set<int>("Block Size",nev/2);
    pls.set<int>("Maximum Subspace Dimension",3*nev);
    printer->stream(Warnings) << "Testing solver(nev/2,3*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,false,istate,false);
    pls.set<int>("Maximum Subspace Dimension",3*nev);
    printer->stream(Warnings) << "Testing solver(nev/2,3*nev) with generalized eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probgen,printer,orthogen,sorter,pls,false,istate,false);

    pls.set<int>("Block Size",nev/2);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(nev/2,4*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,false,istate,false);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(nev/2,4*nev) with generalized eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probgen,printer,orthogen,sorter,pls,false,istate,false);

    // try with an invalid blockSize
    pls.set<int>("Block Size",0);
    pls.set<int>("Maximum Subspace Dimension",4*nev);
    printer->stream(Warnings) << "Testing solver(0,4*nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,true,istate,false);

    // try with an invalid maxdim
    pls.set<int>("Block Size",nev);
    pls.set<int>("Maximum Subspace Dimension",0);
    printer->stream(Warnings) << "Testing solver(nev,0) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,true,istate,false);

    // try with an invalid maxdim: invalid because it must be greater than nev
    pls.set<int>("Block Size",nev);
    pls.set<int>("Maximum Subspace Dimension",nev);
    printer->stream(Warnings) << "Testing solver(nev,nev) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,true,istate,false);

    // try with a too-large subspace
    probstd->setHermitian(true);
    probstd->setProblem();
    pls.set<int>("Maximum Subspace Dimension",100+1);
    printer->stream(Warnings) << "Testing solver(4,toomany,Hermitian) with standard eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,true,istate,false);

    // try with an unset problem
    // setHermitian will mark the problem as unset
    probstd->setHermitian(true);
    printer->stream(Warnings) << "Testing solver with unset eigenproblem..." << std::endl;
    testsolver<ST,MT,MVT,MV,OP>(probstd,printer,orthostd,sorter,pls,true,istate,false);
    // set problem: now hermitian
    probstd->setProblem();

    // create a dummy status tester
    RCP< StatusTest<ST,MV,OP> > dumtester = rcp( new StatusTestMaxIters<ST,MV,OP>(1) );

    // try with a null problem
    printer->stream(Warnings) << "Testing solver with null eigenproblem..." << std::endl;
    try {
      RCP< GeneralizedDavidson<ST,MV,OP> > solver
        = rcp( new GeneralizedDavidson<ST,MV,OP>(Teuchos::null,sorter,printer,dumtester,orthostd,pls) );
      TEUCHOS_TEST_FOR_EXCEPTION(true,get_out,"Instantiating with invalid parameters failed to throw exception.");
    }
    catch (const std::invalid_argument &ia) {
      // caught expected exception
    }

    // try with a null sortman
    printer->stream(Warnings) << "Testing solver with null sort manager..." << std::endl;
    try {
      RCP< GeneralizedDavidson<ST,MV,OP> > solver
        = rcp( new GeneralizedDavidson<ST,MV,OP>(probstd,Teuchos::null,printer,dumtester,orthostd,pls) );
      TEUCHOS_TEST_FOR_EXCEPTION(true,get_out,"Instantiating with invalid parameters failed to throw exception.");
    }
    catch (const std::invalid_argument &ia) {
      // caught expected exception
    }

    // try with a output man problem
    printer->stream(Warnings) << "Testing solver with null output manager..." << std::endl;
    try {
      RCP< GeneralizedDavidson<ST,MV,OP> > solver
        = rcp( new GeneralizedDavidson<ST,MV,OP>(probstd,sorter,Teuchos::null,dumtester,orthostd,pls) );
      TEUCHOS_TEST_FOR_EXCEPTION(true,get_out,"Instantiating with invalid parameters failed to throw exception.");
    }
    catch (const std::invalid_argument &ia) {
      // caught expected exception
    }

    // try with a null status test
    printer->stream(Warnings) << "Testing solver with null status test..." << std::endl;
    try {
      RCP< GeneralizedDavidson<ST,MV,OP> > solver
        = rcp( new GeneralizedDavidson<ST,MV,OP>(probstd,sorter,printer,Teuchos::null,orthostd,pls) );
      TEUCHOS_TEST_FOR_EXCEPTION(true,get_out,"Instantiating with invalid parameters failed to throw exception.");
    }
    catch (const std::invalid_argument &ia) {
      // caught expected exception
    }

    // try with a null orthoman
    printer->stream(Warnings) << "Testing solver with null ortho manager..." << std::endl;
    try {
      RCP< GeneralizedDavidson<ST,MV,OP> > solver
        = rcp( new GeneralizedDavidson<ST,MV,OP>(probstd,sorter,printer,dumtester,Teuchos::null,pls) );
      TEUCHOS_TEST_FOR_EXCEPTION(true,get_out,"Instantiating with invalid parameters failed to throw exception.");
    }
    catch (const std::invalid_argument &ia) {
      // caught expected exception
    }

  }
  catch (const get_out &go) {
    printer->stream(Errors) << "Test failed: " << go.what() << std::endl;
    testFailed = true;
  }
  catch (const std::exception &e) {
    printer->stream(Errors) << "Caught unexpected exception: " << e.what() << std::endl;
    testFailed = true;
  }

  if (testFailed) {
    printer->stream(Warnings) << std::endl << "End Result: TEST FAILED" << std::endl;
    return -1;
  }
  //
  // Default return value
  //
  printer->stream(Warnings) << std::endl << "End Result: TEST PASSED" << std::endl;
  return 0;

} // end run

int main(int argc, char *argv[]) {
  return run<double>(argc,argv);
  // run<float>(argc,argv);
}
