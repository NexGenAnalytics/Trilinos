// $Id$
// $Source$

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

#include "Pitchfork_FiniteElementProblem.hpp"

#include <Teuchos_Assert.hpp>

#include "Basis.H"
#include "NOX_Common.H"

// Constructor - creates the Tpetra objects (maps and vectors)
template <typename ScalarType>
Pitchfork_FiniteElementProblem::Pitchfork_FiniteElementProblem(
    int numGlobalElements, Teuchos::RCP<const Teuchos::Comm<int>>& comm)
    : flag(F_ONLY),
      standardMap(NULL),
      overlapMap(NULL),
      importer(NULL),
      initialSolution(NULL),
      rhs(NULL),
      AA(NULL),
      A(NULL),
      comm(&comm),
      rank(0),
      numProcs(0),
      numLocalElements(0),
      numGlobalElements(numGlobalElements),
      lambda(0.0),
      alpha(0.0),
      beta(0.0) {
  // Commonly used variables
  int i;
  rank = comm->getRank();      // Process ID
  numProcs = comm->NumProc();  // Total number of processes

  // Construct a Source Map that puts approximately the same
  // Number of equations on each processor in uniform global ordering
  standardMap = new map_type(numGlobalElements, 0, *comm);

  // Get the number of elements owned by this processor
  numLocalElements = standardMap->getLocalNumElements();

  // Construct an overlaped map for the finite element fill *************
  // For single processor jobs, the overlap and standard map are the same
  if (NumProc == 1) {
    overlapMap = new map_type(*standardMap);
  } else {
    int overlapNumLocalElements;
    int overlapMinMyGID;

    overlapNumLocalElements = numLocalElements + 2;
    if ((myRank == 0) || (myRank == NumProc - 1))
      overlapNumLocalElements--;

    if (myRank == 0) {
      overlapMinMyGID = standardMap->MinMyGID();
    }
    else {
      overlapMinMyGID = standardMap->MinMyGID() - 1;
    }

    int* overlapMyGlobalElements = new int[overlapNumLocalElements];

    for (i = 0; i < overlapNumLocalElements; i++) {
      overlapMyGlobalElements[i] = overlapMinMyGID + i;
    } 

    overlapMap = new map_type(-1, overlapNumLocalElements, OverlapMyGlobalElements, 0, *comm);

    delete[] overlapMyGlobalElements;

  }  // End Overlap map construction *************************************

  // Construct Linear Objects
  importer = new import_type(*overlapMap, *standardMap);
  initialSolution = new vector_type(*standardMap);
  AA = new csr_graph_type(Copy, *standardMap, 5);

  // Allocate the memory for a matrix dynamically (i.e. the graph is dynamic).
  generateGraph(*AA);

  // Create a second matrix using graph of first matrix - this creates a
  // static graph so we can refill the new matirx after FillComplete()
  // is called.
  A = new csr_matrix_type(Copy, *AA);
  A->fillComplete();

  // Set default bifurcation values
  lambda = -2.25;
  alpha = 1.0;
  beta = 0.0;
}

// Destructor
template <typename ScalarType>
Pitchfork_FiniteElementProblem::~Pitchfork_FiniteElementProblem() {
  delete AA;
  delete A;
  delete initialSolution;
  delete importer;
  delete overlapMap;
  delete standardMap;
}

// Matrix and Residual Fills
template <typename ScalarType>
bool Pitchfork_FiniteElementProblem::evaluate(FillType f, const Tpetra_Vector* soln,
                                              Tpetra_Vector* tmp_rhs, Epetra_RowMatrix* tmp_matrix,
                                              double jac_coeff, double mass_coeff) {
  flag = f;

  // Set the incoming linear objects
  if (flag == F_ONLY) {
    rhs = tmp_rhs;
  } else if (flag == MATRIX_ONLY) {
    A = dynamic_cast<Tpetra_CrsMatrix*>(tmp_matrix);
    assert(A != NULL);
  } else if (flag == ALL) {
    rhs = tmp_rhs;
    A = dynamic_cast<Tpetra_CrsMatrix*>(tmp_matrix);
    assert(A != NULL);
  } else {
    std::cout << "ERROR: Pitchfork_FiniteElementProblem::fillMatrix() - FillType flag is broken"
              << std::endl;
    throw;
  }

  // Create the overlapped solution and position vectors
  Tpetra_Vector u(*overlapMap);
  Tpetra_Vector x(*overlapMap);

  // Export Solution to Overlap vector
  u.Import(*soln, *Importer, Insert);

  // Declare required variables
  int i, j, ierr;
  int overlapNumLocalElements = overlapMap->NumMyElements();

  int overlapMinMyGID;
  if (myRank == 0)
    overlapMinMyGID = standardMap->MinMyGID();
  else
    overlapMinMyGID = standardMap->MinMyGID() - 1;

  int row, column;
  double jac;
  double xx[2];
  double uu[2];
  Basis basis;

  // Create the nodal coordinates
  double Length = 2.0;
  double dx = Length / ((double)NumGlobalElements - 1);
  for (i = 0; i < overlapNumLocalElements; i++) {
    x[i] = -1.0 + dx * ((double)overlapMinMyGID + i);
  }

  // Zero out the objects that will be filled
  if ((flag == MATRIX_ONLY) || (flag == ALL)) {
    i = A->PutScalar(0.0);
    assert(i == 0);
  }
  if ((flag == F_ONLY) || (flag == ALL)) {
    i = rhs->PutScalar(0.0);
    assert(i == 0);
  }

  // Loop Over # of Finite Elements on Processor
  for (int ne = 0; ne < overlapNumLocalElements - 1; ne++) {
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
        row = overlapMap->GID(ne + i);
        // printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
        //      myRank, row, ne+i,standardMap.MyGID(row));
        if (standardMap->MyGID(row)) {
          if ((flag == F_ONLY) || (flag == ALL)) {
            (*rhs)[standardMap->LID(overlapMap->GID(ne + i))] +=
                +basis.wt * basis.dx *
                ((-1.0 / (basis.dx * basis.dx)) * basis.duu * basis.dphide[i] -
                 source_term(basis.uu) * basis.phi[i]);
          }
        }
        // Loop over Trial Functions
        if ((flag == MATRIX_ONLY) || (flag == ALL)) {
          for (j = 0; j < 2; j++) {
            if (standardMap->MyGID(row)) {
              column = overlapMap->GID(ne + j);
              jac = jac_coeff * basis.wt * basis.dx *
                        ((-1.0 / (basis.dx * basis.dx)) * basis.dphide[j] * basis.dphide[i] -
                         source_deriv(basis.uu) * basis.phi[j] * basis.phi[i]) +
                    mass_coeff * basis.wt * basis.dx * basis.phi[j] * basis.phi[i];
              ierr = A->SumIntoGlobalValues(row, 1, &jac, &column);
              TEUCHOS_ASSERT(ierr == 0);
            }
          }
        }
      }
    }
  }

  // Insert Boundary Conditions and modify Jacobian and function (F)
  // U(-1)=beta
  if (myRank == 0) {
    if ((flag == F_ONLY) || (flag == ALL))
      (*rhs)[0] = (*soln)[0] - beta;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      column = 0;
      jac = 1.0 * jac_coeff;
      A->ReplaceGlobalValues(0, 1, &jac, &column);
      column = 1;
      jac = 0.0 * jac_coeff;
      A->ReplaceGlobalValues(0, 1, &jac, &column);
    }
  }

  // U(1)=beta
  if (standardMap->getLocalElement(standardMap->getMaxGlobalIndex()) >= 0) {
    int lastDof = standardMap->getLocalElement(standardMap->getMaxGlobalIndex());
    if ((flag == F_ONLY) || (flag == ALL))
      (*rhs)[lastDof] = (*soln)[lastDof] - beta;
    if ((flag == MATRIX_ONLY) || (flag == ALL)) {
      int row = standardMap->getMaxGlobalIndex();
      column = row;
      jac = 1.0 * jac_coeff;
      A->replaceGlobalValues(row, 1, &jac, &column);
      column = row - 1;
      jac = 0.0 * jac_coeff;
      A->replaceGlobalValues(row, 1, &jac, &column);
    }
  }
  // Sync up processors to be safe
  comm->barrier();

  A->fillComplete();

  return true;
}

Tpetra_Vector& Pitchfork_FiniteElementProblem::getSolution() { return *initialSolution; }

Tpetra_CrsMatrix& Pitchfork_FiniteElementProblem::getJacobian() { return *A; }

bool Pitchfork_FiniteElementProblem::setParameter(std::string label, double value) {
  if (label == "lambda")
    lambda = value;
  else if (label == "alpha")
    alpha = value;
  else if (label == "beta")
    beta = value;
  else if (label == "Homotopy Continuation Parameter") {
    // do nothing for now
  } else {
    // do nothing (may be a constraint parameter that we don't know about)
  }
  return true;
}

tcsrgraph_t& Pitchfork_FiniteElementProblem::generateGraph(tcsrgraph_t& AAA) {
  // Declare required variables
  int i, j;
  int row, column;
  int overlapNumLocalElements = overlapMap->NumMyElements();

  // Loop Over # of Finite Elements on Processor
  for (int ne = 0; ne < overlapNumLocalElements - 1; ne++) {
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
  AAA.FillComplete();
  return AAA;
}

double Pitchfork_FiniteElementProblem::source_term(double x) {
  return lambda * x - alpha * x * x + beta * x * x * x;
}

double Pitchfork_FiniteElementProblem::source_deriv(double x) {
  return lambda - 2.0 * alpha * x + 3.0 * beta * x * x;
}
