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

#ifndef NOX_TPETRA_FINITEDIFFERENCE_H
#define NOX_TPETRA_FINITEDIFFERENCE_H

#include "NOX_Tpetra_Interface_Jacobian.H"          // base class
#include "NOX_Tpetra_Interface_Preconditioner.H"    // base class

#include "NOX_Epetra_Interface_Required.H" // for enum FillType
#include "NOX_Common.H"                    // for <string>
#include "Teuchos_RCP.hpp"
#include "TVector.h"

// Forward Declarations;
class TMap;
class TImport;
class TVector;
class TCrsGraph;
class TCrsMatrix;

namespace NOX {
  namespace Abstract {
    class Group;
  }
  namespace Tpetra {
    class Vector;
  }
}

namespace NOX {

namespace Tpetra {

  /*! \brief Concrete implementation for creating an TRowMatrix Jacobian via finite differencing of the residual.

The Jacobian entries are calculated via 1st order finite differencing.
This requires \f$ N + 1 \f$ calls to computeF() where \f$ N \f$ is the
number of unknowns in the problem.

  \f[ J_{ij} = \frac{\partial F_i}{\partial x_j} = \frac{F_i(x+\delta\mathbf{e}_j) - F_i(x)}{\delta}  \f]

where \f$ J\f$ is the Jacobian, \f$ F\f$ is the function evaluation,
\f$ x\f$ is the solution vector, and \f$\delta\f$ is a small
perturbation to the \f$ x_j\f$ entry.

The perturbation, \f$ \delta \f$, is calculated based on one of the
following equations:

\f[ \delta = \alpha * | x_j | + \beta \f]
\f[ \delta = \alpha * | x_j | + \beta_j \f]

where \f$ \alpha \f$ is a scalar value (defaults to 1.0e-4) and \f$
\beta \f$ can be either a scalar or a vector (defaults to a scalar
value of 1.0e-6).  The choice is defined by the type of constructor
used.  All parameters are supplied in the constructor.  In addition to
the forward difference derivative approximation, backward or centered
differences can be used via the setDifferenceMethod function.  Note
that centered difference provides second order spatial accuracy but at
the cost of twice as many function evaluations.

  Since this inherits from the TRowMatrix class, it can be used as the preconditioning matrix for AztecOO preconditioners.  This method is very inefficient when computing the Jacobian and is not recommended for large-scale systems but only for debugging purposes.
  */
class FiniteDifference : public TRowMatrix,
             public NOX::Tpetra::Interface::Jacobian,
             public NOX::Tpetra::Interface::Preconditioner {

 public:

  //! Define types for use of the perturbation parameter \f$ \delta\f$.
  enum DifferenceType {Forward, Backward, Centered};

  //! Constructor with scalar beta.
  FiniteDifference(Teuchos::ParameterList& printingParams,
                   const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
           const NOX::Tpetra::Vector& initialGuess,
           double beta = 1.0e-6,
           double alpha = 1.0e-4);

  //! Constructor with vector beta.
  FiniteDifference(Teuchos::ParameterList& printingParams,
                   const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
           const NOX::Tpetra::Vector& initialGuess,
           const Teuchos::RCP<const TVector>& beta,
           double alpha = 1.0e-4);

  //! Constructor that takes a pre-constructed TCrsGraph so it does not have to determine the non-zero entries in the matrix.
  FiniteDifference(Teuchos::ParameterList& printingParams,
                   const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
           const NOX::Tpetra::Vector& initialGuess,
           const Teuchos::RCP<TCrsGraph>& g,
           double beta = 1.0e-6,
           double alpha = 1.0e-4);

  //! Constructor with output control that takes a pre-constructed TCrsGraph so it does not have to determine the non-zero entries in the matrix.
  FiniteDifference(Teuchos::ParameterList& printingParams,
                   const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
           const NOX::Tpetra::Vector& initialGuess,
           const Teuchos::RCP<TCrsGraph>& g,
           const Teuchos::RCP<const TVector>& beta,
           double alpha = 1.0e-4);

  //! Pure virtual destructor
  virtual ~FiniteDifference();

  //! Returns a character std::string describing the name of the operator
  virtual const char* Label () const;

  //! If set true, the transpose of this operator will be applied
  virtual int SetUseTranspose(bool UseTranspose);

  //! Return the result on an Epetra_Operator applied to an TMultiVector X in Y.
  virtual int Apply(const TMultiVector& X, TMultiVector& Y) const;

  //! Return the result on an Epetra_Operator inverse applied to an TMultiVector X in Y.
  virtual int ApplyInverse(const TMultiVector& X, TMultiVector& Y) const;

  //! Returns the current use transpose setting
  virtual bool UseTranspose() const;

  //! Returns true if the this object can provide an approximate Inf-norm, false otherwise.
  virtual bool HasNormInf() const;

  //!Returns the TBlockMap object associated with the domain of this matrix operator.
  virtual const TMap & OperatorDomainMap() const;

  //!Returns the TBlockMap object associated with the range of this matrix operator.
  virtual const TMap & OperatorRangeMap() const;

  //! See TRowMatrix documentation.
  virtual bool Filled() const;

  //! See TRowMatrix documentation.
  virtual int NumMyRowEntries(int MyRow, int & NumEntries) const;

  //! See TRowMatrix documentation.
  virtual int MaxNumEntries() const;

  //! See TRowMatrix documentation.
  virtual int ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const;

  //! See TRowMatrix documentation.
  virtual int ExtractDiagonalCopy(TVector & Diagonal) const;

  //! See TRowMatrix documentation.
  virtual int Multiply(bool TransA, const TMultiVector& X, TMultiVector& Y) const;

  //! See TRowMatrix documentation.
  virtual int Solve(bool Upper, bool Trans, bool UnitDiagonal, const TMultiVector& X,  TMultiVector& Y) const;

  //! See TRowMatrix documentation.
  virtual int InvRowSums(TVector& x) const;

  //! See TRowMatrix documentation.
  virtual int LeftScale(const TVector& x);

  //! See TRowMatrix documentation.
  virtual int InvColSums(TVector& x) const;

  //! See TRowMatrix documentation.
  virtual int RightScale(const TVector& x);

  //! See TRowMatrix documentation.
  virtual double NormInf() const;

  //! See TRowMatrix documentation.
  virtual double NormOne() const;

  //! See TRowMatrix documentation.
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
   virtual int NumGlobalNonzeros() const;
#endif
   virtual long long NumGlobalNonzeros64() const;

  //! See TRowMatrix documentation.
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
   virtual int NumGlobalRows() const;
#endif
   virtual long long NumGlobalRows64() const;

  //! See TRowMatrix documentation.
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
   virtual int NumGlobalCols() const;
#endif
   virtual long long NumGlobalCols64() const;

  //! See TRowMatrix documentation.
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
   virtual int NumGlobalDiagonals() const;
#endif
   virtual long long NumGlobalDiagonals64() const;

  //! See TRowMatrix documentation.
  virtual int NumMyNonzeros() const;

  //! See TRowMatrix documentation.
  virtual int NumMyRows() const;

  //! See TRowMatrix documentation.
  virtual int NumMyCols() const;

  //! See TRowMatrix documentation.
  virtual int NumMyDiagonals() const;

  //! See TRowMatrix documentation.
  virtual bool LowerTriangular() const;

  //! See TRowMatrix documentation.
  virtual bool UpperTriangular() const;

  //! See TRowMatrix documentation.
  virtual const Teuchos::RCP<const Teuchos::Comm<int>>& comm() const;

  //! See TRowMatrix documentation.
  virtual const TMap & RowMatrixRowMap() const;

  //! See TRowMatrix documentation.
  virtual const TMap & RowMatrixColMap() const;

  //! See TRowMatrix documentation.
  virtual const TImport * RowMatrixImporter() const;

  //! See Epetra_SrcDistObj documentation.
  virtual const TBlockMap& Map() const;

  //! Compute Jacobian given the specified input vector, x. Returns true if computation was successful.
  virtual bool computeJacobian(const TVector& x, TOperator& Jac);

  //! Compute Jacobian given the specified input vector, x. Returns true if computation was successful.
  virtual bool computeJacobian(const TVector& x);

  //! Compute an TRowMatrix to be used by Aztec preconditioners given the specified input vector, x. Returns true if computation was successful.
  virtual bool computePreconditioner(const TVector& x,
                     TOperator& Prec,
                     Teuchos::ParameterList* precParams = 0);

  //! Set the type of perturbation method used (default is Forward)
  virtual void setDifferenceMethod( DifferenceType type );

  //! An accessor method for the underlying TCrsMatrix
  virtual TCrsMatrix& getUnderlyingMatrix() const;

  //! Output the underlying matrix
  virtual void Print(std::ostream&) const;

  //! Register a NOX::Abstract::Group derived object and use the computeF() method of that group for the perturbation instead of the NOX::Tpetra::Interface::Required::computeF() method.  This is required for LOCA to get the operators correct during homotopy.
  void setGroupForComputeF(NOX::Abstract::Group& group);

protected:

  //! Constructs an TCrsGraph and TRowMatrix for the Jacobian.  This is only called if the user does not supply an TCrsGraph.
  Teuchos::RCP<TCrsMatrix>
  createGraphAndJacobian(Interface::Required& i, const TVector& x);

  bool computeF(const TVector& input, TVector& result,
        NOX::Tpetra::Interface::Required::FillType);

protected:

  //! Printing Utilities object
  const NOX::Utils utils;

  //! Pointer to the Jacobian graph.
  Teuchos::RCP<TCrsGraph> graph;

  //! Pointer to the Jacobian.
  Teuchos::RCP<TCrsMatrix> jacobian;

  //! User provided interface function.
  Teuchos::RCP<NOX::Tpetra::Interface::Required> interface;

  //! Perturbed solution vector - a work array that needs to be mutable.
  mutable TVector x_perturb;

  //! Function evaluation at currentX - a work array that needs to be mutable.
  mutable TVector fo;

  //! Function evaluation at perturbX - a work array that needs to be mutable.
  mutable TVector fp;

  //! Optional pointer to function evaluation at -perturbX - needed only for centered finite differencing
  Teuchos::RCP<TVector> fmPtr;

  //! Column vector of the jacobian - a work array that needs to be mutable.
  mutable TVector Jc;

  //! Constant for the perturbation calculation.
  double alpha;

  //! Constant for the perturbation calculation.
  double beta;

  //! Vector for the perturbation calculation.
  Teuchos::RCP<const TVector> betaVector;

  //! Define types for the \f$ \beta \f$ parameter during the
  //! computation of the perturbation parameter \f$ \delta\f$.
  enum BetaType {Scalar, Vector};

  //! Flag that sets whether \f$ \beta \f$ is a scalar or a vector.
  BetaType betaType;

  //! Define types for use of the perturbation parameter \f$ \delta\f$.
  DifferenceType diffType;

  //! label for the TRowMatrix
  std::string label;

  //! Flag to enables the use of a group instead of the interface for the computeF() calls in the directional difference calculation.
  bool useGroupForComputeF;

  //! Pointer to the group for possible use in computeF() calls.
  Teuchos::RCP<NOX::Abstract::Group> groupPtr;

};
}  // namespace Tpetra
}  // namespace NOX

#endif /* NOX_TPETRA_FINITEDIFFERENCE_H */
