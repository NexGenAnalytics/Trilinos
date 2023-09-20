//@HEADER
// ************************************************************************
//
//            LOCA: Library of Continuation Algorithms Package
//                 Copyright (2005) Sandia Corporation
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

#include <NOX_Common.H>

#include "Basis.hpp"
#include "FiniteElementProblem.hpp"

// Constructor - creates the Tpetra objects (maps and vectors)
template typename<ST>
FiniteElementProblem<ST>::FiniteElementProblem(int numGlobalElements,
                       Teuchos::RCP<const Teuchos::Comm<int>> &comm,
                       ST s) :
  Comm(&comm),
  NumGlobalElements(numGlobalElements),
  scale(s)
{

  // Commonly used variables
  int i;
  MyPID = Comm->getRank();      // Process ID
  NumProc = Comm->getSize();  // Total number of processes

  // Construct a Source Map that puts approximately the same
  // Number of equations on each processor in uniform global ordering
  StandardMap = new tmap_t(NumGlobalElements, 0, *Comm);

  // Get the number of elements owned by this processor
  NumMyElements = StandardMap->getLocalNumElements();

  // Construct an overlaped map for the finite element fill *************
  // For single processor jobs, the overlap and standard map are the same
  if (NumProc == 1) {
    OverlapMap = new tmap_t(*StandardMap);
  } else {

    int OverlapNumMyElements;
    int OverlapMinMyGID;
    OverlapNumMyElements = NumMyElements + 2;
    if ((MyPID == 0) || (MyPID == NumProc - 1))
      OverlapNumMyElements --;

    if (MyPID==0)
      OverlapMinMyGID = StandardMap->getMinGlobalIndex();
    else
      OverlapMinMyGID = StandardMap->getMinGlobalIndex() - 1;

    int* OverlapMyGlobalElements = new int[OverlapNumMyElements];

    for (i = 0; i < OverlapNumMyElements; i ++)
      OverlapMyGlobalElements[i] = OverlapMinMyGID + i;

    OverlapMap = new tmap_t(-1, OverlapNumMyElements,
                OverlapMyGlobalElements, 0, *Comm);

  } // End Overlap map construction *************************************

  // Construct Linear Objects
  Importer = new timport_t(*OverlapMap, *StandardMap);
  initialSolution = new tvector_t(*StandardMap);
  AA = new tcrsgraph_t(Copy, *StandardMap, 5);

  // Allocate the memory for a matrix dynamically (i.e. the graph is dynamic).
  generateGraph(*AA);

  // Create a second matrix using graph of first matrix - this creates a
  // static graph so we can refill the new matirx after FillComplete()
  // is called.
  A = new tcrsmatrix_t(Copy, *AA);
  A->FillComplete();

  // Set default bifurcation values
  factor = 0.1;
  leftBC = 1.0;
  rightBC = 1.0;
}

// Matrix and Residual Fills
template typename<ST>
bool FiniteElementProblem::evaluate(FillType f,
                    const tvector_t* soln,
                    tvector_t* tmp_rhs,
                    trowmatrix_t* tmp_matrix)
{
  flag = f;

  // Set the incoming linear objects
  if (flag == F_ONLY) {
    rhs = tmp_rhs;
  } else if (flag == MATRIX_ONLY) {
    A = dynamic_cast<tcrsmatrix_t*> (tmp_matrix);
  } else if (flag == ALL) {
    rhs = tmp_rhs;
    A = dynamic_cast<tcrsmatrix_t*> (tmp_matrix);
  } else {
    std::cout << "ERROR: FiniteElementProblem::fillMatrix() - FillType flag is broken" << std::endl;
    throw;
  }

  // Create the overlapped solution and position vectors
  tvector_t u(*OverlapMap);
  tvector_t x(*OverlapMap);

  // Export Solution to Overlap vector
  u.doImport(*soln, *Importer, Tpetra::INSERT);

  // Declare required variables
  int i,j,ierr;
  int OverlapNumMyElements = OverlapMap->NumMyElements();

  int OverlapMinMyGID;
  if (MyPID==0) OverlapMinMyGID = StandardMap->getMinGlobalIndex();
  else OverlapMinMyGID = StandardMap->getMinGlobalIndex()-1;

  int row, column;
  ST jac;
  ST xx[2];
  ST uu[2];
  Basis basis;

  // Create the nodal coordinates
  ST Length=1.0;
  ST dx=Length/((ST) NumGlobalElements-1);
  for (i=0; i < OverlapNumMyElements; i++) {
    x[i]=dx*((ST) OverlapMinMyGID+i);
  }

  // Zero out the objects that will be filled
  if ((flag == MATRIX_ONLY) || (flag == ALL)) i=A->PutScalar(0.0);
  if ((flag == F_ONLY)    || (flag == ALL)) i=rhs->PutScalar(0.0);

  // Loop Over # of Finite Elements on Processor
  for (int ne=0; ne < OverlapNumMyElements-1; ne++) {

    // Loop Over Gauss Points
    for(int gp=0; gp < 2; gp++) {
      // Get the solution and coordinates at the nodes
      xx[0]=x[ne];
      xx[1]=x[ne+1];
      uu[0]=u[ne];
      uu[1]=u[ne+1];
      // Calculate the basis function at the gauss point
      basis.getBasis(gp, xx, uu);

      // Loop over Nodes in Element
      for (i=0; i< 2; i++) {
    row=OverlapMap->GID(ne+i);
    //printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
    //     MyPID, row, ne+i,StandardMap.MyGID(row));
    if (StandardMap->MyGID(row)) {
      if ((flag == F_ONLY)    || (flag == ALL)) {
        (*rhs)[StandardMap->LID(OverlapMap->GID(ne+i))]+=
          +basis.wt*basis.dx
          *((-1.0/(basis.dx*basis.dx))*basis.duu*
        basis.dphide[i]+factor*basis.uu*basis.uu*basis.uu*basis.phi[i]);
      }
    }
    // Loop over Trial Functions
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      for(j=0;j < 2; j++) {
        if (StandardMap->MyGID(row)) {
          column=OverlapMap->GID(ne+j);
          jac=basis.wt*basis.dx*((-1.0/(basis.dx*basis.dx))*
                     basis.dphide[j]*basis.dphide[i]
                     +3.0*factor*basis.uu*basis.uu*basis.phi[j]*
                     basis.phi[i]);
          ierr=A->sumIntoGlobalValues(row, 1, &jac, &column);
        }
      }
    }
      }
    }
  }

  // Insert Boundary Conditions and modify Jacobian and function (F)
  // U(0)=1
  if (MyPID==0) {
    if ((flag == F_ONLY)    || (flag == ALL))
      (*rhs)[0]= (*soln)[0] - leftBC;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      column=0;
      jac=1.0;
      A->replaceGlobalValues(0, 1, &jac, &column);
      column=1;
      jac=0.0;
      A->replaceGlobalValues(0, 1, &jac, &column);
    }
  }

  if ( StandardMap->LID(StandardMap->getMaxAllGlobalIndex()) >= 0 ) {
    int lastDof = StandardMap->LID(StandardMap->getMaxAllGlobalIndex());
    if ((flag == F_ONLY)    || (flag == ALL))
       (*rhs)[lastDof] = (*soln)[lastDof] - rightBC;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
       int row = StandardMap->getMaxAllGlobalIndex();
       column = row;
      jac=1.0;
      A->replaceGlobalValues(row, 1, &jac, &column);
      column=row-1;
      jac=0.0;
      A->replaceGlobalValues(row, 1, &jac, &column);
    }
  }
  // Sync up processors to be safe
  Comm->barrier();

  A->fillComplete();

  return true;
}

template typename<ST>
Teuchos::RCP<tvector_t> FiniteElementProblem::getSolution()
{
  return initialSolution;
}

template typename<ST>
Teuchos::RCP<tcrsmatrix_t> FiniteElementProblem::getJacobian()
{
  return A;
}

template typename<ST>
bool FiniteElementProblem::setParameter(std::string label, ST value)
{
  if (label == "Nonlinear Factor")
    factor = value;
  else if (label == "Left BC")
    leftBC = value;
  else if (label == "Right BC")
    rightBC = value*scale;
  else if (label == "Homotopy Continuation Parameter") {
    // do nothing for now
  }
  else {
    std::cout << "ERROR: FiniteElementProblem::setParameter() - label is invalid "
     << "for this problem!" << std::endl;
    exit(-1);
  }
  return true;
}

template typename<ST>
tcrsgraph_t& FiniteElementProblem::generateGraph(tcrsgraph_t& AAA)
{
  // Declare required variables
  int i,j;
  int row, column;
  int OverlapNumMyElements = OverlapMap->getLocalNumElements();
  int OverlapMinMyGID;
  if (MyPID==0) OverlapMinMyGID = StandardMap->getMinGlobalIndex();
  else OverlapMinMyGID = StandardMap->getMinGlobalIndex()-1;

  // Loop Over # of Finite Elements on Processor
  for (int ne=0; ne < OverlapNumMyElements-1; ne++) {

    // Loop over Nodes in Element
    for (i=0; i< 2; i++) {
      row=OverlapMap->GID(ne+i);

      // Loop over Trial Functions
      for(j=0;j < 2; j++) {
        
        // If this row is owned by current processor, add the index
        if (StandardMap->MyGID(row)) {
          column=OverlapMap->GID(ne+j);
          AAA.insertGlobalIndices(row, 1, &column);
        }
      }
    }
  }
  AAA.fillComplete();
  return AAA;
}
