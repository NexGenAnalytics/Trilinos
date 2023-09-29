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
template <typename Scalar>
FiniteElementProblem<ST>::FiniteElementProblem(int numGlobalElements,
                                               Teuchos::RCP<const Teuchos::Comm<int>>& comm,
                                               Scalar s)
    : comm(&comm), numGlobalElements(numGlobalElements), scale(s) {
  // Commonly used variables
  int i;
  myRank = comm->getRank();   // Process ID
  numProc = comm->getSize();  // Total number of processes

  // Construct a Source Map that puts approximately the same
  // Number of equations on each processor in uniform global ordering
  standardMap = new map_type(numGlobalElements, 0, *comm);

  // Get the number of elements owned by this processor
  numMyElements = standardMap->getLocalNumElements();

  // Construct an overlaped map for the finite element fill *************
  // For single processor jobs, the overlap and standard map are the same
  if (numProc == 1) {
    overlapMap = new map_type(*standardMap);
  } else {
    int overlapNumMyElements;
    int overlapMinMyGID;
    overlapNumMyElements = numMyElements + 2;
    if ((myRank == 0) || (myRank == numProc - 1))
      overlapNumMyElements--;

    if (myRank == 0) {
      overlapMinMyGID = standardMap->getMinGlobalIndex();
    } else {
      overlapMinMyGID = standardMap->getMinGlobalIndex() - 1;
    }

    int* overlapMyGlobalElements = new int[overlapNumMyElements];

    for (i = 0; i < overlapNumMyElements; i++) {
      overlapMyGlobalElements[i] = overlapMinMyGID + i;
    }

    overlapMap = new tmap_t(-1, overlapNumMyElements, overlapMyGlobalElements, 0, *comm);

  }  // End Overlap map construction *************************************

  // Construct Linear Objects
  importer = new timport_t(*overlapMap, *standardMap);
  initialSolution = new tvector_t(*standardMap);
  AA = new tcrsgraph_t(Copy, *standardMap, 5);

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
template <typename Scalar>
bool FiniteElementProblem::evaluate(FillType f, const tvector_t* soln, tvector_t* tmp_rhs,
                                    trowmatrix_t* tmp_matrix) {
  flag = f;

  // Set the incoming linear objects
  if (flag == F_ONLY) {
    rhs = tmp_rhs;
  } else if (flag == MATRIX_ONLY) {
    A = dynamic_cast<tcrsmatrix_t*>(tmp_matrix);
  } else if (flag == ALL) {
    rhs = tmp_rhs;
    A = dynamic_cast<tcrsmatrix_t*>(tmp_matrix);
  } else {
    std::cout << "ERROR: FiniteElementProblem::fillMatrix() - FillType flag is broken" << std::endl;
    throw;
  }

  // Create the overlapped solution and position vectors
  vector_type u(*overlapMap);
  vector_type x(*overlapMap);

  // Export Solution to Overlap vector
  u.doImport(*soln, *importer, Tpetra::INSERT);

  // Declare required variables
  int i, j, ierr;
  int overlapNumMyElements = overlapMap->numMyElements();

  int overlapMinMyGID;
  if (myRank == 0)
    overlapMinMyGID = standardMap->getMinGlobalIndex();
  else
    overlapMinMyGID = standardMap->getMinGlobalIndex() - 1;

  int row, column;
  ST jac;
  ST xx[2];
  ST uu[2];
  Basis basis;

  // Create the nodal coordinates
  ST Length = 1.0;
  ST dx = Length / ((ST)numGlobalElements - 1);
  for (i = 0; i < overlapNumMyElements; i++) {
    x[i] = dx * ((ST)overlapMinMyGID + i);
  }

  // Zero out the objects that will be filled
  if ((flag == MATRIX_ONLY) || (flag == ALL))
    i = A->PutScalar(0.0);
  if ((flag == F_ONLY) || (flag == ALL))
    i = rhs->PutScalar(0.0);

  // Loop Over # of Finite Elements on Processor
  for (int ne = 0; ne < overlapNumMyElements - 1; ne++) {
    // Loop Over Gauss Points
    for (int gp = 0; gp < 2; gp++) {
      // Get the solution and coordinates at the nodes
      xx[0] = x[ne];
      xx[1] = x[ne + 1];
      uu[0] = u[ne];
      uu[1] = u[ne + 1];
      // Calculate the basis function at the gauss point
      basis.getBasis(gp, xx, uu);

      // Loop over Nodes in Element
      for (i = 0; i < 2; i++) {
        row = overlapMap->getGlobalElement(ne + i);
        // printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
        //      myRank, row, ne+i,standardMap.MyGID(row));
        if (standardMap->MyGID(row)) {
          if ((flag == F_ONLY) || (flag == ALL)) {
            (*rhs)[standardMap->LID(overlapMap->GID(ne + i))] +=
                +basis.wt * basis.dx *
                ((-1.0 / (basis.dx * basis.dx)) * basis.duu * basis.dphide[i] +
                 factor * basis.uu * basis.uu * basis.uu * basis.phi[i]);
          }
        }
        // Loop over Trial Functions
        if ((flag == MATRIX_ONLY) || (flag == ALL)) {
          for (j = 0; j < 2; j++) {
            if (standardMap->getLocalElement(row)) {
              column = overlapMap->GID(ne + j);
              jac = basis.wt * basis.dx *
                    ((-1.0 / (basis.dx * basis.dx)) * basis.dphide[j] * basis.dphide[i] +
                     3.0 * factor * basis.uu * basis.uu * basis.phi[j] * basis.phi[i]);
              ierr = A->sumIntoGlobalValues(row, 1, &jac, &column);
            }
          }
        }
      }
    }
  }

  // Insert Boundary Conditions and modify Jacobian and function (F)
  // U(0)=1
  if (myRank == 0) {
    if ((flag == F_ONLY) || (flag == ALL))
      (*rhs)[0] = (*soln)[0] - leftBC;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      column = 0;
      jac = 1.0;
      A->replaceGlobalValues(0, 1, &jac, &column);
      column = 1;
      jac = 0.0;
      A->replaceGlobalValues(0, 1, &jac, &column);
    }
  }

  if (standardMap->LID(standardMap->getMaxAllGlobalIndex()) >= 0) {
    int lastDof = standardMap->getLocalElement(standardMap->getMaxAllGlobalIndex());
    if ((flag == F_ONLY) || (flag == ALL))
      (*rhs)[lastDof] = (*soln)[lastDof] - rightBC;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      int row = standardMap->getMaxAllGlobalIndex();
      column = row;
      jac = 1.0;
      A->replaceGlobalValues(row, 1, &jac, &column);
      column = row - 1;
      jac = 0.0;
      A->replaceGlobalValues(row, 1, &jac, &column);
    }
  }
  // Sync up processors to be safe
  comm->barrier();

  A->fillComplete();

  return true;
}

template <typename Scalar>
Teuchos::RCP<tvector_t> FiniteElementProblem::getSolution() {
  return initialSolution;
}

template <typename Scalar>
Teuchos::RCP<tcrsmatrix_t> FiniteElementProblem::getJacobian() {
  return A;
}

template <typename Scalar>
bool FiniteElementProblem::setParameter(std::string label, ST value) {
  if (label == "Nonlinear Factor")
    factor = value;
  else if (label == "Left BC")
    leftBC = value;
  else if (label == "Right BC")
    rightBC = value * scale;
  else if (label == "Homotopy Continuation Parameter") {
    // do nothing for now
  } else {
    std::cout << "ERROR: FiniteElementProblem::setParameter() - label is invalid "
              << "for this problem!" << std::endl;
    exit(-1);
  }
  return true;
}

template <typename Scalar>
tcrsgraph_t& FiniteElementProblem::generateGraph(tcrsgraph_t& AAA) {
  // Declare required variables
  int i, j;
  int row, column;
  int overlapNumMyElements = overlapMap->getLocalNumElements();
  int overlapMinMyGID;
  if (myRank == 0)
    overlapMinMyGID = standardMap->getMinGlobalIndex();
  else
    overlapMinMyGID = standardMap->getMinGlobalIndex() - 1;

  // Loop Over # of Finite Elements on Processor
  for (int ne = 0; ne < overlapNumMyElements - 1; ne++) {
    // Loop over Nodes in Element
    for (i = 0; i < 2; i++) {
      row = overlapMap->GID(ne + i);

      // Loop over Trial Functions
      for (j = 0; j < 2; j++) {
        // If this row is owned by current processor, add the index
        if (standardMap->MyGID(row)) {
          column = overlapMap->GID(ne + j);
          AAA.insertGlobalIndices(row, 1, &column);
        }
      }
    }
  }
  AAA.fillComplete();
  return AAA;
}
