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

// Finite Element Problem Class
/* Provides function (F) and Jacobian evaluations for the following equation
 * via a 1D linear finite element discretization with Epetra objects.
 *
 * d2u
 * --- - k * u**2 = 0
 * dx2
 *
 * subject to @ x=0, u=1
 */

#ifndef _NOX_EXAMPLE_TPETRA_NONLINEAR_FINITEELEMENTPROBLEM_HPP
#define _NOX_EXAMPLE_TPETRA_NONLINEAR_FINITEELEMENTPROBLEM_HPP

#include <Teuchos_Comm.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_CombineMode.hpp>

#include "Basis.hpp"

// Flag to tell the evaluate routine what objects to fill
enum FillType { F_ONLY, MATRIX_ONLY, ALL };

// Flag to tell the evaluate routine how to ghost shared information
enum OverlapType { NODES, ELEMENTS };

// Finite Element Problem Class
template<typename ScalarType>
class Brusselator
{
  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NO = typename Tpetra::Map<>::node_type;

  using tvector_t     = typename Tpetra::Vector<ST, LO, GO>;
  using tmap_t        = typename Tpetra::Map<LO, GO, NO>;
  using tcrsmatrix_t  = typename Tpetra::CrsMatrix<ST, LO, GO>;
  using tcrsgraph_t   = typename Tpetra::CrsGraph<LO, GO, NO>;
  using timport_t     = typename Tpetra::Import<LO, GO, NO>;

public:
  // Constructor
  Brusselator(int NumGlobalUnknowns_,
              Teuchos::RCP< const Teuchos::Comm<int> > comm,
              OverlapType OType_ = ELEMENTS) :
    xmin(0.0),
    xmax(1.0),
    dt(5.0e-1),
    flag(F_ONLY),
    overlapType(OType_),
    ColumnToOverlapImporter(0),
    rhs(NULL),
    Comm(comm),
    NumGlobalNodes(NumGlobalUnknowns_)
  {
    // Commonly used variables
    int i;
    MyPID = Comm->getRank();     // Process ID
    NumProc = Comm->getSize();  // Total number of processes

    // Here we assume a 2-species Brusselator model, ie 2 dofs per node
    // Note that this needs to be echoed in thew anonymous enum for NUMSPECIES.
    NumSpecies = 2;
    NumGlobalUnknowns = NumSpecies * NumGlobalNodes;

    // Construct a Source Map that puts approximately the same
    // number of equations on each processor

    // Begin by distributing nodes fairly equally
    StandardNodeMap = Teuchos::rcp(new tmap_t(NumGlobalNodes, 0, Comm));

    // Get the number of nodes owned by this processor
    NumMyNodes = StandardNodeMap->getLocalNumElements();

    // Construct an overlap node map for the finite element fill
    // For single processor jobs, the overlap and standard maps are the same
    if (NumProc == 1)
      OverlapNodeMap = Teuchos::rcp(new tmap_t(*StandardNodeMap));
    else {

      if( overlapType == ELEMENTS ) {

        int OverlapNumMyNodes;
        int OverlapMinMyNodeGID;

        OverlapNumMyNodes = NumMyNodes + 2;
        if ((MyPID == 0) || (MyPID == NumProc - 1))
          OverlapNumMyNodes --;

        if (MyPID==0)
          OverlapMinMyNodeGID = StandardNodeMap->getMinGlobalIndex();
        else
          OverlapMinMyNodeGID = StandardNodeMap->getMinGlobalIndex() - 1;

        int* OverlapMyGlobalNodes = new int[OverlapNumMyNodes];

        for (i = 0; i < OverlapNumMyNodes; i ++)
          OverlapMyGlobalNodes[i] = OverlapMinMyNodeGID + i;

        OverlapNodeMap = Teuchos::rcp(new tmap_t(-1, &OverlapNumMyNodes, *OverlapMyGlobalNodes, 0, Comm));
      }

      else { // overlapType == NODES

        // Here we distribute elements such that nodes are ghosted, i.e. no
        // overlapping elements but instead overlapping nodes
        int OverlapNumMyNodes;
        int OverlapMinMyNodeGID;

        OverlapNumMyNodes = NumMyNodes + 1;
        if ( MyPID == NumProc - 1 )
          OverlapNumMyNodes --;

        OverlapMinMyNodeGID = StandardNodeMap->getMinGlobalIndex();

        int* OverlapMyGlobalNodes = new int[OverlapNumMyNodes];

          for (i = 0; i < OverlapNumMyNodes; i ++)
          OverlapMyGlobalNodes[i] = OverlapMinMyNodeGID + i;

        OverlapNodeMap = Teuchos::rcp(new tmap_t(-1, &OverlapNumMyNodes, *OverlapMyGlobalNodes, 0, Comm));
      }
    } // End Overlap node map construction ********************************

    // Now create the unknowns maps from the node maps
    NumMyUnknowns = NumSpecies * NumMyNodes;
    int* StandardMyGlobalUnknowns = new int[NumMyUnknowns];
    for (int k=0; k<NumSpecies; k++)
      for (i=0; i<NumMyNodes; i++)

        // For now, we employ an interleave of unknowns

        StandardMyGlobalUnknowns[ NumSpecies * i + k ] =
                NumSpecies * StandardNodeMap->getGlobalElement(i) + k;

    StandardMap = Teuchos::rcp(new tmap_t(-1, &NumMyUnknowns, *StandardMyGlobalUnknowns, 0, Comm));

    assert(StandardMap->getGlobalNumElements() == NumGlobalUnknowns);

    if (NumProc == 1) {
      OverlapMap = Teuchos::rcp(new tmap_t(*StandardMap));
    }
    else {
      int OverlapNumMyNodes = OverlapNodeMap->getLocalNumElements ();
      int OverlapNumMyUnknowns = NumSpecies * OverlapNumMyNodes;
      int* OverlapMyGlobalUnknowns = new int[OverlapNumMyUnknowns];
      for (int k=0; k<NumSpecies; k++)
        for (i=0; i<OverlapNumMyNodes; i++)
          OverlapMyGlobalUnknowns[ NumSpecies * i + k ] =
                  NumSpecies * OverlapNodeMap->getGlobalElement(i) + k;

      OverlapMap = Teuchos::rcp(new tmap_t(-1, &OverlapNumMyUnknowns, *OverlapMyGlobalUnknowns, 0, Comm));
    } // End Overlap unknowns map construction ***************************


  #ifdef DEBUG_MAPS
    // Output to check progress so far
    printf("[%d] NumSpecies, NumMyNodes, NumGlobalNodes, NumMyUnknowns, NumGlobalUnknowns :\n %d\t%d\t%d\t%d\n",
      MyPID,NumSpecies, NumMyNodes, NumGlobalNodes, NumMyUnknowns, NumGlobalUnknowns);

    Comm->Barrier();
    Comm->Barrier();
    Comm->Barrier();
    printf("[%d] StandardNodeMap :\n", MyPID);
    Comm->Barrier();
    std::cout << *StandardNodeMap << std::endl;
    Comm->Barrier();
    printf("[%d] OverlapNodeMap :\n", MyPID);
    Comm->Barrier();
    std::cout << *OverlapNodeMap << std::endl;
    Comm->Barrier();
    printf("[%d] StandardMap :\n", MyPID);
    Comm->Barrier();
    std::cout << *StandardMap << std::endl;
    Comm->Barrier();
    printf("[%d] StandardMap :\n", MyPID);
    Comm->Barrier();
    std::cout << *OverlapMap << std::endl;
    Comm->Barrier();
  #endif

    // Construct Linear Objects
    Importer = Teuchos::rcp(new timport_t(*OverlapMap, *StandardMap));
    nodeImporter = Teuchos::rcp(new timport_t(*OverlapNodeMap, *StandardNodeMap));
    initialSolution = Teuchos::rcp(new tvector_t(*StandardMap));
    oldSolution = new tvector_t(*StandardMap);
    AA = Teuchos::rcp(new tcrsgraph_t(Teuchos::Copy, *StandardMap, 0));

    // Allocate the memory for a matrix dynamically (i.e. the graph is dynamic).
    if( overlapType == NODES )
      generateGraphUsingNodes(*AA);
    else
      generateGraphUsingElements(*AA);

  #ifdef DEBUG_GRAPH
    AA->Print(cout);
  #endif
  #ifdef DEBUG_IMPORTER
    Importer->Print(cout);
  #endif

    // Create the Importer needed for FD coloring using element overlap
    // as well as the problem Jacobian matrix which may or may not be used
    // depending on the choice of Jacobian operator
    if( overlapType == ELEMENTS ) {
      ColumnToOverlapImporter = new timport_t(AA->getColMap(),*OverlapMap);
      A = Teuchos::rcp(new tcrsmatrix_t(Teuchos::Copy, *AA));
      A->FillComplete();
    }

    // Create the nodal coordinates
    xptr = Teuchos::rcp(new tvector_t(*StandardNodeMap));
    ST Length= xmax - xmin;
    dx=Length/((ST) NumGlobalNodes-1);
    for (i=0; i < NumMyNodes; i++) {
      (*xptr)[i]=xmin + dx*((ST) StandardNodeMap->getMinGlobalIndex()+i);
    }

    initializeSoln();
    
  }

  // Reset problem for next parameter (time) step.
  // For now, this simply updates oldsoln with the given Tpetra_Vector
  void reset(const tvector_t x)
  {
    *oldSolution = x;
  }

  // Set initial condition for solution vector
  void initializeSoln()
  {
    tvector_t& soln = *initialSolution;
    tvector_t& x = *xptr;

    // Here we do a sinusoidal perturbation of the unstable
    // steady state.

    ST pi = 4.*atan(1.0);

    for (int i=0; i<x.MyLength(); i++) {
      soln[2*i] = 0.6 + 1.e-1*sin(1.0*pi*x[i]);
      soln[2*i+1] = 10.0/3.0 + 1.e-1*sin(1.0*pi*x[i]);
    }
    *oldSolution = soln;
  }

  // Evaluates the function (F) and/or the Jacobian using the solution
  // values in solnVector.
  bool evaluate(/*NOX::Epetra::Interface::Required::FillType fType,*/ // AM: TOFIX
                const tvector_t *soln,
                tvector_t *tmp_rhs,
                tvector_t *tmp_matrix)
  {
    /*if( fType == Tpetra::Jac ) {*/    // AM: TOFIX
      if( overlapType == NODES ) {
        std::cout << "This problem only works for Finite-Difference Based Jacobians"
            << std::endl << "when overlapping nodes." << std::endl
            << "No analytic Jacobian fill available !!" << std::endl;
        exit(1);
      }
      flag = MATRIX_ONLY;
    /*} else {*/      // AM: TOFIX
      flag = F_ONLY;
      if( overlapType == NODES ){
        rhs = new tvector_t(*OverlapMap);
      }else{
        rhs = tmp_rhs;
      }
    /*}*/   // AM: TOFIX

    // Create the overlapped solution and position vectors
    tvector_t u(*OverlapMap);
    tvector_t uold(*OverlapMap);
    tvector_t xvec(*OverlapNodeMap);

    // Export Solution to Overlap vector
    // If the vector to be used in the fill is already in the Overlap form,
    // we simply need to map on-processor from column-space indices to
    // OverlapMap indices. Note that the old solution is simply fixed data that
    // needs to be sent to an OverlapMap (ghosted) vector.  The conditional
    // treatment for the current soution vector arises from use of
    // FD coloring in parallel.
    uold.doImport(*oldSolution, Importer, Tpetra::INSERT);
    xvec.doImport(*xptr, *nodeImporter, Tpetra::INSERT);
    if( (overlapType == ELEMENTS) /*&& (fType == Tpetra::FD_Res)*/)               // AM: TOFIX
      // Overlap vector for solution received from FD coloring, so simply reorder
      // on processor
      u.Export(*soln, *ColumnToOverlapImporter, Tpetra::INSERT);
    else // Communication to Overlap vector is needed
      u.doImport(*soln, *Importer, Tpetra::INSERT);

    // Declare required variables
    int i,j;
    int OverlapNumMyNodes = OverlapNodeMap->getLocalNumElements();

    int row1, row2, column1, column2;
    ST term1, term2;
    ST Dcoeff1 = 0.025;
    ST Dcoeff2 = 0.025;
    ST alpha = 0.6;
    ST beta = 2.0;
    ST jac11, jac12, jac21, jac22;
    ST xx[2];
    ST uu[2*NUMSPECIES] = {0.0}; // Use of the anonymous enum is needed for SGI builds
    ST uuold[2*NUMSPECIES] = {0.0};
    Basis basis(NumSpecies);

    // Zero out the objects that will be filled
    if ((flag == MATRIX_ONLY) || (flag == ALL)) A->PutScalar(0.0);
    if ((flag == F_ONLY)      || (flag == ALL)) rhs->PutScalar(0.0);

    // Loop Over # of Finite Elements on Processor
    for (int ne=0; ne < OverlapNumMyNodes - 1; ne++) {

      // Loop Over Gauss Points
      for(int gp=0; gp < 2; gp++) {
        // Get the solution and coordinates at the nodes
        xx[0]=xvec[ne];
        xx[1]=xvec[ne+1];
        for (int k=0; k<NumSpecies; k++) {
          uu[NumSpecies * 0 + k] = u[NumSpecies * ne + k];
          uu[NumSpecies * 1 + k] = u[NumSpecies * (ne+1) + k];
          uuold[NumSpecies * 0 + k] = uold[NumSpecies * ne + k];
          uuold[NumSpecies * 1 + k] = uold[NumSpecies * (ne+1) + k];
        }
        // Calculate the basis function at the gauss point
        basis.getBasis(gp, xx, uu, uuold);

        // Loop over Nodes in Element
        for (i=0; i< 2; i++) {
          row1=OverlapMap->getGlobalElement(NumSpecies * (ne+i));
          row2=OverlapMap->getGlobalElement(NumSpecies * (ne+i) + 1);
          term1 = basis.wt*basis.dx
              *((basis.uu[0] - basis.uuold[0])/dt * basis.phi[i]
                +(1.0/(basis.dx*basis.dx))*Dcoeff1*basis.duu[0]*basis.dphide[i]
                + basis.phi[i] * ( -alpha + (beta+1.0)*basis.uu[0]
                - basis.uu[0]*basis.uu[0]*basis.uu[1]) );
          term2 = basis.wt*basis.dx
                *((basis.uu[1] - basis.uuold[1])/dt * basis.phi[i]
                +(1.0/(basis.dx*basis.dx))*Dcoeff2*basis.duu[1]*basis.dphide[i]
                + basis.phi[i] * ( -beta*basis.uu[0]
                + basis.uu[0]*basis.uu[0]*basis.uu[1]) );
      //printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
      //     MyPID, row, ne+i,StandardMap.MyGID(row));
          if ((flag == F_ONLY)    || (flag == ALL)) {
            if( overlapType == NODES ) {
              (*rhs)[NumSpecies*(ne+i)]   += term1;
          (*rhs)[NumSpecies*(ne+i)+1] += term2;
            }
        else
          if (StandardMap->isNodeLocalElement(row1)) {
                (*rhs)[StandardMap->getLocalElement (OverlapMap->getGlobalElement(NumSpecies*(ne+i)))]+=
                  term1;
                (*rhs)[StandardMap->getLocalElement (OverlapMap->getGlobalElement(NumSpecies*(ne+i)+1))]+=
                  term2;
        }
      }
          // Loop over Trial Functions
          if ((flag == MATRIX_ONLY) || (flag == ALL)) {
            for(j=0;j < 2; j++) {
              if (StandardMap->isNodeLocalElement(row1)) {
                column1=OverlapMap->getGlobalElement(NumSpecies * (ne+j));
                column2=OverlapMap->getGlobalElement(NumSpecies * (ne+j) + 1);
                jac11=basis.wt*basis.dx*(
                        basis.phi[j]/dt*basis.phi[i]
                        +(1.0/(basis.dx*basis.dx))*Dcoeff1*basis.dphide[j]*
                                                          basis.dphide[i]
                        + basis.phi[i] * ( (beta+1.0)*basis.phi[j]
                        - 2.0*basis.uu[0]*basis.phi[j]*basis.uu[1]) );
                jac12=basis.wt*basis.dx*(
                        basis.phi[i] * ( -basis.uu[0]*basis.uu[0]*basis.phi[j]) );
                jac21=basis.wt*basis.dx*(
                        basis.phi[i] * ( -beta*basis.phi[j]
                        + 2.0*basis.uu[0]*basis.phi[j]*basis.uu[1]) );
                jac22=basis.wt*basis.dx*(
                        basis.phi[j]/dt*basis.phi[i]
                        +(1.0/(basis.dx*basis.dx))*Dcoeff2*basis.dphide[j]*
                                                          basis.dphide[i]
                        + basis.phi[i] * basis.uu[0]*basis.uu[0]*basis.phi[j] );
                A->SumIntoGlobalValues(row1, 1, &jac11, &column1);
                column1++;
                A->SumIntoGlobalValues(row1, 1, &jac12, &column1);
                A->SumIntoGlobalValues(row2, 1, &jac22, &column2);
                column2--;
                A->SumIntoGlobalValues(row2, 1, &jac21, &column2);
              }
            }
          }
        }
      }
    }

    // Insert Boundary Conditions and modify Jacobian and function (F)
    // U(0)=1
    if (MyPID==0) {
      if ((flag == F_ONLY)    || (flag == ALL)) {
        (*rhs)[0]= (*soln)[0] - 0.6;
        (*rhs)[1]= (*soln)[1] - 10.0/3.0;
      }
      if ((flag == MATRIX_ONLY) || (flag == ALL)) {
        int column=0;
        ST jac=1.0;
        A->ReplaceGlobalValues(0, 1, &jac, &column);
        column++;
        A->ReplaceGlobalValues(1, 1, &jac, &column);
        jac=0.0;
        column=0;
        A->ReplaceGlobalValues(1, 1, &jac, &column);
        column++;
        A->ReplaceGlobalValues(0, 1, &jac, &column);
        column++;
        A->ReplaceGlobalValues(0, 1, &jac, &column);
        A->ReplaceGlobalValues(1, 1, &jac, &column);
        column++;
        A->ReplaceGlobalValues(0, 1, &jac, &column);
        A->ReplaceGlobalValues(1, 1, &jac, &column);
      }
    }
    // U(1)=1
    if ( StandardMap->getLocalElement (StandardMap->getMaxAllGlobalIndex()) >= 0 ) {
      int lastDof = StandardMap->getLocalElement (StandardMap->getMaxAllGlobalIndex());
      if ((flag == F_ONLY)    || (flag == ALL)) {
        (*rhs)[lastDof - 1] = (*soln)[lastDof - 1] - 0.6;
        (*rhs)[lastDof] = (*soln)[lastDof] - 10.0/3.0;
      }
      if ((flag == MATRIX_ONLY) || (flag == ALL)) {
        int row=StandardMap->getMaxAllGlobalIndex() - 1;
        int column = row;
        ST jac = 1.0;
        A->ReplaceGlobalValues(row++, 1, &jac, &column);
        column++;
        A->ReplaceGlobalValues(row, 1, &jac, &column);
        jac=0.0;
        row = column - 1;
        A->ReplaceGlobalValues(row, 1, &jac, &column);
        column--;
        A->ReplaceGlobalValues(row+1, 1, &jac, &column);
        column--;
        A->ReplaceGlobalValues(row, 1, &jac, &column);
        A->ReplaceGlobalValues(row+1, 1, &jac, &column);
        column--;
        A->ReplaceGlobalValues(row, 1, &jac, &column);
        A->ReplaceGlobalValues(row+1, 1, &jac, &column);
      }
    }

    // Sync up processors to be safe
    Comm->barrier();

    // Do an assemble for overlap nodes
    if( overlapType == NODES )
      tmp_rhs->Export(*rhs, *Importer, Tpetra::ADD);

    //  Comm->Barrier();
    //  std::cout << "Returned tmp_rhs residual vector :\n" << std::endl << *tmp_rhs << std::endl;
    
    return true;
  }

  // Return a reference to the Epetra_Vector with the initial guess
  // that is generated by the Brusselator class.
  Teuchos::RCP<tvector_t> getSolution()
  {
    return initialSolution;
  }

  // Return a reference to the Epetra_Vector with the Jacobian
  // that is generated by the Brusselator class.
  Teuchos::RCP<tcrsmatrix_t> getJacobian()
  {
    if( Teuchos::is_null(A) ) return A;
    else {
      std::cout << "No valid Jacobian matrix for this problem. This is likely the"
                << " result of overlapping NODES rather than ELEMENTS.\n"
                << std::endl;
      throw "Brusselator Error";
    }
  }
  
  // Return a reference to the Epetra_Vector with the mesh positions
  Teuchos::RCP<tvector_t> getMesh()
  {
    return xptr;
  }

  // Accesor function for time step
  ST getdt()
  {
    return dt;
  }
  
  // Return a reference to the Epetra_Vector with the old solution
  tvector_t& getOldSoln()
  {
    return *oldSolution;
  }

  // Return a reference to the Epetra_CrsGraph that is generated by
  // the Brusselator problem class.
  Teuchos::RCP<tcrsgraph_t> getGraph()
  {
    return AA;
  }
  
private:
  
  // inserts the global column indices into the Graph
  tcrsgraph_t& generateGraphUsingNodes(tcrsgraph_t& AA)
  {
    int row, column;

    int myMinNodeGID = StandardNodeMap->getMinLocalIndex();
    int myMaxNodeGID = StandardNodeMap->getMaxLocalIndex();

    int leftNodeGID, rightNodeGID;
    for( int myNode = myMinNodeGID; myNode <= myMaxNodeGID; myNode++ ) {

      leftNodeGID  = myNode - 1;
      rightNodeGID = myNode + 1;

      if( leftNodeGID < StandardNodeMap->getMinAllGlobalIndex() )
        leftNodeGID = StandardNodeMap->getMinAllGlobalIndex();

      if( rightNodeGID > StandardNodeMap->getMaxAllGlobalIndex() )
        rightNodeGID = StandardNodeMap->getMaxAllGlobalIndex();

      for( int dependNode = leftNodeGID; dependNode <= rightNodeGID; dependNode++ ) {

        // Loop over unknowns in Node
        for (int j = 0; j < NumSpecies; j++) {
          row = NumSpecies * myNode + j; // Interleave scheme

          // Loop over unknowns at supporting nodes
          for (int m = 0; m < NumSpecies; m++) {
            column = NumSpecies * dependNode + m;
            //printf("\t\tWould like to insert -> (%d, %d)\n",row,column);
            AA.insertGlobalIndices(row, 1, &column);
          }
        }
      }
    }
    AA.fillComplete();
    return AA;
  }

  // inserts the global column indices into the Graph
  tcrsgraph_t& generateGraphUsingElements(tcrsgraph_t& AA)
  {
    // Declare required variables
    int i,j;
    int row, column;
    int OverlapNumMyNodes = OverlapNodeMap->getLocalNumElements();

    // Loop Over # of Finite Elements on Processor
    for (int ne=0; ne < OverlapNumMyNodes-1; ne++) {

      // Loop over Nodes in Element
      for (i=0; i<2; i++) {

        // If this node is owned by current processor, add indices
        if (StandardNodeMap->isNodeLocalElement (OverlapNodeMap->getGlobalElement(ne+i))) {

          // Loop over unknowns in Node
          for (int k=0; k<NumSpecies; k++) {
            row=OverlapMap->getGlobalElement( NumSpecies*(ne+i) + k); // Interleave scheme

            // Loop over supporting nodes
            for(j=0; j<2; j++) {

              // Loop over unknowns at supporting nodes
              for (int m=0; m<NumSpecies; m++) {
                column=OverlapMap->getGlobalElement( NumSpecies*(ne+j) + m);
                //printf("\t\tWould like to insert -> (%d, %d)\n",row,column);
                AA.insertGlobalIndices(row, 1, &column);
              }
            }
          }
        }
      }
    }
    AA.fillComplete();
    return AA;
  }
  
private:
  ST xmin;
  ST xmax;
  ST dx;
  ST dt;

  FillType flag;
  OverlapType overlapType;
  
  Teuchos::RCP<tmap_t> StandardNodeMap;
  Teuchos::RCP<tmap_t> OverlapNodeMap;
  Teuchos::RCP<tmap_t> StandardMap;
  Teuchos::RCP<tmap_t> OverlapMap;
  Teuchos::RCP<timport_t> Importer;
  Teuchos::RCP<timport_t> nodeImporter;
  Teuchos::RCP<timport_t> ColumnToOverlapImporter;
  Teuchos::RCP<tvector_t> xptr;
  Teuchos::RCP<tvector_t> initialSolution;
  tvector_t *oldSolution;
  tvector_t *rhs;
  Teuchos::RCP<tcrsgraph_t> AA;
  Teuchos::RCP<tcrsmatrix_t> A;
  Teuchos::RCP< const Teuchos::Comm<int> > Comm;

  int MyPID;              // Process number

  int NumProc;            // Total number of processes
  int NumSpecies;         // Number of unknowns owned by this process
  int NumMyNodes;         // Number of nodes owned by this process
  int NumGlobalNodes;     // Total Number of nodes
  int NumMyUnknowns;      // Number of unknowns owned by this process
  int NumGlobalUnknowns;  // Total Number of unknowns
  
  enum { NUMSPECIES = 2}; // Needed for SGI build
};
#endif
