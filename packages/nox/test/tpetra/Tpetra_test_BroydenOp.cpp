//@HEADER
// ************************************************************************
//
//            NOX: An Object-Oriented Nonlinear Solver Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov) or
// Eric Phipps (etphipp@sandia.gov), Sandia National Laboratories.
// ************************************************************************
//  CVS Information
//  $Source$
//  $Author$
//  $Date$
//  $Revision$
// ************************************************************************
//@HEADER

// NOX headers
#include <NOX.H>
#include "NOX_TestCompare.H" // Test Suite headers
// #include "NOX_Epetra_DebugTools.H" // CWS: necessary?

// Trilinos headers
#include <Tpetra_Map.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_RowMatrix.hpp>

#include "Teuchos_StandardCatchMacros.hpp"


template <typename ScalarType>
int run(int argc, char *argv[]) {

  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using tmap_t       = typename Tpetra::Map<LO,GO,NT>;
  using tvector_t    = typename Tpetra::Vector<ST,LO,GO,NT>;
  using tcrsgraph_t  = typename Tpetra::CrsGraph<LO,GO,NT>;
  using tcrsmatrix_t = typename Tpetra::CrsMatrix<ST,LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;

  // Initialize MPI
  Teuchos::GlobalMPISession session(&argc, &argv, nullptr);

  bool success = false;
  bool verbose = false;

  try {

    const auto Comm = Tpetra::getDefaultComm();

    int * testInt = new int[100];
    delete [] testInt;

    if (argc > 1)
      if (argv[1][0]=='-' && argv[1][1]=='v')
        verbose = true;

    // Get the process ID and the total number of processors
    int MyPID = Comm->getRank();
    int NumProc = Comm->getSize();

    // Set up the solver options parameter list
    Teuchos::RCP<Teuchos::ParameterList> noxParamsPtr = Teuchos::rcp(new Teuchos::ParameterList);
    Teuchos::ParameterList & noxParams = *(noxParamsPtr.get());

    // Set up the printing utilities
    // Only print output if the "-v" flag is set on the command line
    Teuchos::ParameterList& printParams = noxParams.sublist("Printing");
    printParams.set("MyPID", MyPID);
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

    Teuchos::RCP<NOX::Utils> printing = Teuchos::rcp( new NOX::Utils(printParams) );

    // Identify the test problem
    if (printing->isPrintType(NOX::Utils::TestDetails))
      printing->out() << "Starting tpetra/NOX_Tpetra_BroydenOp.exe" << std::endl;

    // Identify processor information
#ifdef HAVE_MPI
    if (printing->isPrintType(NOX::Utils::TestDetails))
    {
      printing->out() << "Parallel Run" << std::endl;
      printing->out() << "Number of processors = " << NumProc << std::endl;
      printing->out() << "Print Process = " << MyPID << std::endl;
    }
    Comm->barrier();
    if (printing->isPrintType(NOX::Utils::TestDetails))
      printing->out() << "Process " << MyPID << " is alive!" << std::endl;
    Comm->barrier();
#else
    if (printing->isPrintType(NOX::Utils::TestDetails))
      printing->out() << "Serial Run" << std::endl;
#endif

    int status = 0;

    // Create a TestCompare class
    NOX::TestCompare tester( printing->out(), *printing);
    ST abstol = 1.e-4;
    ST reltol = 1.e-4 ;

    // Test NOX::Epetra::BroydenOperator
    int numGlobalElems = 3 * NumProc;
    RCP<tmap_t> broydenRowMap = rcp( new tmap_t(numGlobalElems, 0, Comm) );
    RCP<tvector_t> broydenWorkVec = rcp ( new tvector_t(broydenRowMap));
    RCP<tcrsgraph_t> broydenWorkGraph = rcp ( new tcrsgraph_t(broydenRowMap, 0) );

    const std::vector<GO> globalIndices(3);
    for( int lcol = 0; lcol < 3; ++lcol )
      globalIndices[lcol] = 3 * MyPID + lcol;

    const std::vector<GO> myGlobalIndices(2);

    // Row 1 structure
    myGlobalIndices[0] = globalIndices[0];
    myGlobalIndices[1] = globalIndices[2];
    broydenWorkGraph->insertGlobalIndices( globalIndices[0], 2, myGlobalIndices.data() );
    // Row 2 structure
    myGlobalIndices[0] = globalIndices[0];
    myGlobalIndices[1] = globalIndices[1];
    broydenWorkGraph->insertGlobalIndices( globalIndices[1], 2, myGlobalIndices.data() );
    // Row 3 structure
    myGlobalIndices[0] = globalIndices[1];
    myGlobalIndices[1] = globalIndices[2];
    broydenWorkGraph->insertGlobalIndices( globalIndices[2], 2, myGlobalIndices.data() );

    broydenWorkGraph->fillComplete();

    Teuchos::RCP<tcrsmatrix_t> broydenWorkMatrix =
      Teuchos::rcp( new tcrsmatrix_t( broydenWorkGraph ) );

    // Create an identity matrix
    broydenWorkVec.putScalar(1.0);
    broydenWorkMatrix->replaceDiagonalValues(broydenWorkVec);

    NOX::Epetra::BroydenOperator broydenOp( noxParams, printing, broydenWorkVec, broydenWorkMatrix, true );

    broydenWorkVec[0] =  1.0;
    broydenWorkVec[1] = -1.0;
    broydenWorkVec[2] =  2.0;
    broydenOp.setStepVector( broydenWorkVec );

    broydenWorkVec[0] =  2.0;
    broydenWorkVec[1] =  1.0;
    broydenWorkVec[2] =  3.0;
    broydenOp.setYieldVector( broydenWorkVec );

    broydenOp.computeSparseBroydenUpdate();

    // Create the gold matrix for comparison
    Teuchos::RCP<tcrsmatrix_t> goldMatrix = Teuchos::rcp( new tcrsmatrix_t( Teuchos::Copy, broydenWorkGraph ) );

    int      numCols ;
    ST * values  ;

    // Row 1 answers
    goldMatrix->getLocalRowView( 0, numCols, values );
    values[0] =  6.0 ;
    values[1] =  2.0 ;
    // Row 2 answers
    goldMatrix->getLocalRowView( 1, numCols, values );
    values[0] =  5.0 ;
    values[1] =  0.0 ;
    // Row 3 structure
    goldMatrix->getLocalRowView( 2, numCols, values );
    values[0] = -1.0 ;
    values[1] =  7.0 ;

    goldMatrix->Scale(0.2);

    status += tester.testCrsMatrices( broydenOp.getBroydenMatrix(), *goldMatrix, reltol, abstol,
        "Broyden Sparse Operator Update Test" );


    // Now try a dense Broyden Update
    RCP<tcrsgraph_t> broydenWorkGraph2 = rcp ( new tcrsgraph_t(broydenRowMap, 0) );

    myGlobalIndices.resize(3);

    // All Rowsstructure
    myGlobalIndices[0] = globalIndices[0];
    myGlobalIndices[1] = globalIndices[1];
    myGlobalIndices[2] = globalIndices[2];
    broydenWorkGraph2->insertGlobalIndices( globalIndices[0], 3, myGlobalIndices.data() );
    broydenWorkGraph2->insertGlobalIndices( globalIndices[1], 3, myGlobalIndices.data() );
    broydenWorkGraph2->insertGlobalIndices( globalIndices[2], 3, myGlobalIndices.data() );

    broydenWorkGraph2->fillComplete();

    Teuchos::RCP<tcrsmatrix_t> broydenWorkMatrix2 = Teuchos::rcp( new tcrsmatrix_t(broydenWorkGraph2) );

    // Create an identity matrix
    broydenWorkVec.putScalar(1.0);
    broydenWorkMatrix2->replaceDiagonalValues(broydenWorkVec);

    NOX::Epetra::BroydenOperator broydenOp2( noxParams, printing, broydenWorkVec, broydenWorkMatrix2, true );

    broydenWorkVec[0] =  1.0;
    broydenWorkVec[1] = -1.0;
    broydenWorkVec[2] =  2.0;
    broydenOp2.setStepVector( broydenWorkVec );

    broydenWorkVec[0] =  2.0;
    broydenWorkVec[1] =  1.0;
    broydenWorkVec[2] =  3.0;
    broydenOp2.setYieldVector( broydenWorkVec );

    broydenOp2.computeSparseBroydenUpdate();

    // Create the gold matrix for comparison
    Teuchos::RCP<tcrsmatrix_t> goldMatrix2 = Teuchos::rcp( new tcrsmatrix_t( Teuchos::Copy, broydenWorkGraph2 ) );

    // Row 1 answers
    goldMatrix2->getLocalRowView( 0, numCols, values );
    values[0] =  7.0 ;
    values[1] = -1.0 ;
    values[2] =  2.0 ;
    // Row 2 answers
    goldMatrix2->getLocalRowView( 1, numCols, values );
    values[0] =  2.0 ;
    values[1] =  4.0 ;
    values[2] =  4.0 ;
    // Row 3 structure
    goldMatrix2->getLocalRowView( 2, numCols, values );
    values[0] =  1.0 ;
    values[1] = -1.0 ;
    values[2] =  8.0 ;

    ST scaleF = 1.0 / 6.0;
    goldMatrix2->Scale( scaleF );

    status += tester.testCrsMatrices( broydenOp2.getBroydenMatrix(), *goldMatrix2, reltol, abstol,
        "Broyden Sparse Operator Update Test (Dense)" );

    // Now test the ability to remove active entries in the Broyden update
    RCP<tcrsgraph_t> inactiveGraph = rcp ( new tcrsgraph_t(broydenRowMap, 0) );

    // Row 1 structure
    inactiveGraph->insertGlobalIndices( globalIndices[0], 1, myGlobalIndices.data() );
    // Row 2 structure
    inactiveGraph->insertGlobalIndices( globalIndices[1], 1, myGlobalIndices.data() );
    // Row 3 structure
    inactiveGraph->insertGlobalIndices( globalIndices[2], 1, myGlobalIndices.data() );

    inactiveGraph->fillComplete();

    // Inactivate entries in dense matrix to arrive again at the original sparse structure
    broydenOp2.removeEntriesFromBroydenUpdate( inactiveGraph );

#ifdef HAVE_NOX_DEBUG
    if( verbose )
      broydenOp2.outputActiveEntries();
#endif

    // Reset to the identity matrix
    broydenOp2.resetBroydenMatrix( *broydenWorkMatrix2 );

    // Step and Yield vectors are already set
    broydenOp2.computeSparseBroydenUpdate();

    status += tester.testCrsMatrices( broydenOp2.getBroydenMatrix(), *goldMatrix, reltol, abstol,
        "Broyden Sparse Operator Update Test (Entry Removal)", false );

    success = status==0;

    // Summarize test results
    if(success)
      printing->out() << "Test passed!" << std::endl;
    else
      printing->out() << "Test failed!" << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
} // run

int main(int argc, char *argv[]) {
  // templated to run on different ST
  return run<double>(argc,argv);
  // return run<float>(argc,argv);
}

