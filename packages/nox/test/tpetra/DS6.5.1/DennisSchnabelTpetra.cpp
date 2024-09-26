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

// NOX
#include "NOX_Common.H"

#include "DennisSchnabelTpetra.hpp"

// Constructor - creates the Tpetra objects (maps and vectors)
template typename<ST>
DennisSchnabel<ST>::DennisSchnabel(int numGlobalElements, Teuchos::RCP<const Teuchos::Comm<int> > &comm) :
  flag(F_ONLY),
  soln(NULL),
  rhs(NULL),
  Comm(&comm),
  NumGlobalElements(numGlobalElements)
{

  // Commonly used variables
  int i;
  MyPID = Comm->getRank();      // Process ID
  NumProc = Comm->getSize();    // Total number of processes

  // Construct a Source Map that puts approximately the same
  // Number of equations on each processor in uniform global ordering
  StandardMap = new tmap_t(NumGlobalElements, 0, *Comm);

  // Get the number of elements owned by this processor
  LocalNumElements = StandardMap->getLocalNumElements();

  // Construct an overlaped map for the fill calls **********************
  /* The overlap map is needed for multiprocessor jobs.  The unknowns
   * are stored in a distributed vector where each node owns one unknown.
   * During a function or Jacobian evaluation, each processor needs to have
   * both of the unknown values.  The overlapped vector can get this data
   * by importing the owned values from the other node to this overlapped map.
   * Actual solves must be done using the Standard map where everything is
   * distributed.
   */
  // For single processor jobs, the overlap and standard map are the same
  if (NumProc == 1) {
    OverlapMap = new tmap_t(*StandardMap);
  }
  else {

    int OverlapLocalNumElements = 2;
    int OverlapMyGlobalElements[2];

    for (i = 0; i < OverlapLocalNumElements; i ++)
      OverlapMyGlobalElements[i] = i;

    OverlapMap = new tmap_t(-1, OverlapLocalNumElements,
                OverlapMyGlobalElements, 0, *Comm);
  } // End Overlap map construction *************************************

  // Construct Linear Objects
  Importer = new timport_t(*OverlapMap, *StandardMap);
  initialSolution = Teuchos::rcp(new tvector_t(*StandardMap));
  AA = new tcrsgraph_t(Teuchos::Copy, *StandardMap, 5);

  // Allocate the memory for a matrix dynamically (i.e. the graph is dynamic).
  generateGraph(*AA);

  // Use the graph AA to create a Matrix.
  A = Teuchos::rcp(new tcrsmatrix_t (Teuchos::Copy, *AA));

  // Transform the global matrix coordinates to local so the matrix can
  // be operated upon.
  A->fillComplete();
}

// Destructor
template typename<ST>
DennisSchnabel<ST>::~DennisSchnabel()
{
  delete AA;
  delete Importer;
  delete OverlapMap;
  delete StandardMap;
}

// Matrix and Residual Fills
template typename<ST>
bool DennisSchnabel<ST>::evaluate(
             /*NOX::Epetra::Interface::Required::FillType fType,*/ // CWS: check
             const tvector_t* soln,
             tvector_t* tmp_rhs)
{
  flag = MATRIX_ONLY;

  if ( tmp_rhs ) {
    flag = F_ONLY;
    rhs = tmp_rhs;
  }

  // Create the overlapped solution
  tvector_t u(*OverlapMap);

  // Export Solution to Overlap vector so we have all unknowns required
  // for function and Jacobian evaluations.
  u.doImport(*soln, *Importer, Tpetra::INSERT);

  // Begin F fill
  if((flag == F_ONLY) || (flag == ALL)) {

    // Zero out the F vector
    rhs->putScalar(0.0);

    // Processor 0 always fills the first equation.
    if (MyPID==0) {
      (*rhs)[0]=(u[0]*u[0] + u[1]*u[1] - 2.);

      // If it's a single processor job, fill the second equation on proc 0.
      if (NumProc==1)
    (*rhs)[1]=(exp(u[0]-1.) + u[1]*u[1]*u[1] - 2.);
    }
    // Multiprocessor job puts the second equation on processor 1.
    else {
      (*rhs)[0]=(exp(u[0]-1.) + u[1]*u[1]*u[1] - 2.);
    }
  }


  int* column = new int[2];
  double* jac = new double[2];

  // The matrix is 2 x 2 and will always be 0 and 1 regardless of
  // the coordinates being local or global.
  column[0] = 0;
  column[1] = 1;

  // Begin Jacobian fill
  if((flag == MATRIX_ONLY) || (flag == ALL)) {

    // Zero out Jacobian
    A->putScalar(0.0);

    if (MyPID==0) {
      // Processor 0 always fills the first equation.
      jac[0] = 2.*u[0];
      jac[1] = 2.*u[1];
      A->replaceGlobalValues(0, 2, jac, column); // CWS: these won't work initially

      // If it's a single processor job, fill the second equation on proc 0.
      if (NumProc==1) {
    jac[0] = exp(u[0]-1.);
    jac[1] = 3.*u[1]*u[1];
    A->replaceGlobalValues(1, 2, jac, column); // CWS: these won't work initially
      }
    }
    // Multiprocessor job puts the second equation on processor 1.
    else {
      jac[0] = exp(u[0]-1.);
      jac[1] = 3.*u[1]*u[1];
      A->replaceGlobalValues(1, 2, jac, column); // CWS: these won't work initially
    }
  }

  delete [] column;
  delete [] jac;

  // Sync up processors to be safe
  Comm->Barrier();

  // Transform matrix so it can be operated upon.
  A->fillComplete();

  return true;
}

template typename<ST>
Teuchos::RCP<tvector_t> DennisSchnabel<ST>::getSolution()
{
  return initialSolution;
}

template typename<ST>
Teuchos::RCP<tcrsmatrix_t> DennisSchnabel<ST>::getJacobian()
{
  return A;
}

template typename<ST>
tcrsgraph_t& DennisSchnabel::generateGraph<ST>(tcrsgraph_t& AA)
{

  int* index = new int[2];

  if (MyPID==0) {
    index[0]=0;
    index[1]=1;
    AA.insertGlobalIndices(0, 2, index); // CWS: these will not work at first

    if (NumProc==1) {
      index[0]=0;
      index[1]=1;
      AA.insertGlobalIndices(1, 2, index); // CWS: these will not work at first
    }
  } else {
    index[0]=0;
    index[1]=1;
    AA.insertGlobalIndices(1, 2, index); // CWS: these will not work at first
  }

  delete [] index;

  AA.fillComplete();
//   AA.SortIndices();
//   AA.RemoveRedundantIndices();
  return AA;
}
