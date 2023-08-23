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
#include "NOX.H"  // Required headers
// #include "NOX_Epetra.H" // Epetra Interface headers
// #include "NOX_TestCompare.H" // Test Suite headers
// #include "NOX_Epetra_DebugTools.H"

// // Trilinos headers
// #ifdef HAVE_MPI
// #include "Epetra_MpiComm.h"
// #else
// #include "Epetra_SerialComm.h"
// #endif
// #include "TMap.h"
// #include "TVector.h"
// #include "TRowMatrix.h"
// #include "TCrsMatrix.h"
// #include "TMap.h"
// #include "TLinearProblem.h"
// #include "AztecOO.h"
// #include "Teuchos_StandardCatchMacros.hpp"

#include "Laplace2D_Tpetra.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"

int main(int argc, char *argv[])
{
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  bool verbose = false;
  bool success = false;
  try {
    if (argc > 1)
      if (argv[1][0]=='-' && argv[1][1]=='v')
        verbose = true;

    // Get the process ID and the total number of processors
    int MyPID = Comm.MyPID();
#ifdef HAVE_MPI
    int NumProc = Comm.NumProc();
#endif

    // define the parameters of the nonlinear PDE problem
    int nx = 5;
    int ny = 6;
    double lambda = 1.0;

    PDEProblem Problem(nx,ny,lambda,&Comm);

    // starting solution, here a zero vector
    TVector InitialGuess(Problem.GetMatrix()->Map());
    InitialGuess.PutScalar(0.0);

    // random vector upon which to apply each operator being tested
    TVector directionVec(Problem.GetMatrix()->Map());
    directionVec.Random();

    // Set up the problem interface
    Teuchos::RCP<SimpleProblemInterface> interface =
      Teuchos::rcp(new SimpleProblemInterface(&Problem) );

    // Set up theolver options parameter list
    Teuchos::RCP<Teuchos::ParameterList> noxParamsPtr = Teuchos::rcp(new Teuchos::ParameterList);
    Teuchos::ParameterList & noxParams = *(noxParamsPtr.get());

    // Set the nonlinear solver method
    noxParams.set("Nonlinear Solver", "Line Search Based");

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

    NOX::Utils printing(printParams);


    // Identify the test problem
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Starting epetra/NOX_Operators/NOX_Operators.exe" << std::endl;

    // Identify processor information
#ifdef HAVE_MPI
    if (printing.isPrintType(NOX::Utils::TestDetails)) {
      printing.out() << "Parallel Run" << std::endl;
      printing.out() << "Number of processors = " << NumProc << std::endl;
      printing.out() << "Print Process = " << MyPID << std::endl;
    }
    Comm.Barrier();
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Process " << MyPID << " is alive!" << std::endl;
    Comm.Barrier();
#else
    if (printing.isPrintType(NOX::Utils::TestDetails))
      printing.out() << "Serial Run" << std::endl;
#endif

    int status = 0;

    Teuchos::RCP<NOX::Tpetra::Interface::Required> iReq = interface;

    // Need a NOX::Tpetra::Vector for constructor
    TVector noxInitGuess(InitialGuess, NOX::DeepCopy);

    // Analytic matrix
    Teuchos::RCP<TCrsMatrix> A = Teuchos::rcp( Problem.GetMatrix(), false );

    TVector A_resultVec(Problem.GetMatrix()->Map());
    interface->computeJacobian( InitialGuess, *A );
    A->Apply( directionVec, A_resultVec );

    // FD operator
    Teuchos::RCP<TCrsGraph> graph = Teuchos::rcp( const_cast<TCrsGraph*>(&A->graph()), false );
    Teuchos::RCP<NOX::Tpetra::FiniteDifference> FD = Teuchos::rcp(
      new NOX::Tpetra::FiniteDifference(printParams, iReq, noxInitGuess, graph) );

    TVector FD_resultVec(Problem.GetMatrix()->Map());
    FD->computeJacobian(InitialGuess, *FD);
    FD->Apply( directionVec, FD_resultVec );

    // Matrix-Free operator
    Teuchos::RCP<NOX::Tpetra::MatrixFree> MF = Teuchos::rcp(
      new NOX::Tpetra::MatrixFree(printParams, iReq, noxInitGuess) );

    TVector MF_resultVec(Problem.GetMatrix()->Map());
    MF->computeJacobian(InitialGuess, *MF);
    MF->Apply( directionVec, MF_resultVec );

    // Need NOX::Tpetra::Vectors for tests
    NOX::Tpetra::Vector noxAvec ( A_resultVec , NOX::DeepCopy );
    NOX::Tpetra::Vector noxFDvec( FD_resultVec, NOX::DeepCopy );
    NOX::Tpetra::Vector noxMFvec( MF_resultVec, NOX::DeepCopy );

    // Create a TestCompare class
    NOX::Tpetra::TestCompare tester( printing.out(), printing);
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

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
/*
  end of file test.C
*/
