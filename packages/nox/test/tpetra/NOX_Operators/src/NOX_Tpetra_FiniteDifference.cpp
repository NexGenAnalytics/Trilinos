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

#include "NOX_Tpetra_FiniteDifference.hpp"

#include "NOX_Abstract_Group.H"
#include "NOX_Tpetra_Vector.H"
#include "Teuchos_ParameterList.hpp"
#include "NOX_Utils.H"

using namespace NOX;
using namespace NOX::Tpetra;

FiniteDifference::FiniteDifference(
    Teuchos::ParameterList& printingParams,
    const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
    const NOX::Tpetra::Vector& x,
    double beta_,
    double alpha_) :
  utils(printingParams),
  interface(i),
  x_perturb(x.getTpetraVector()),
  fo(x.getTpetraVector()),
  fp(x.getTpetraVector()),
  Jc(x.getTpetraVector()),
  alpha(alpha_),
  beta(beta_),
  betaType(Scalar),
  diffType(Forward),
  label("NOX::FiniteDifference Jacobian"),
  useGroupForComputeF(false)
{
  // Create the finite difference Jacobian matrix
  jacobian = createGraphAndJacobian(*i, x.getTpetraVector());
}

FiniteDifference::
FiniteDifference(
    Teuchos::ParameterList& printingParams,
    const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
    const NOX::Tpetra::Vector& x,
    const Teuchos::RCP<const TVector>& beta_,
    double alpha_) :
  utils(printingParams),
  interface(i),
  x_perturb(x.getTpetraVector()),
  fo(x.getTpetraVector()),
  fp(x.getTpetraVector()),
  Jc(x.getTpetraVector()),
  alpha(alpha_),
  beta(0),
  betaVector(beta_),
  betaType(Vector),
  diffType(Forward),
  label("NOX::FiniteDifference Jacobian"),
  useGroupForComputeF(false)
{
  // Create the finite difference Jacobian matrix
  jacobian = createGraphAndJacobian(*i, x.getTpetraVector());
}

FiniteDifference::FiniteDifference(
    Teuchos::ParameterList& printingParams,
    const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
    const NOX::Tpetra::Vector& x,
    const Teuchos::RCP<TCrsGraph>& userGraph,
    double beta_,
    double alpha_) :
  utils(printingParams),
  graph(userGraph),
  interface(i),
  x_perturb(x.getTpetraVector()),
  fo(x.getTpetraVector()),
  fp(x.getTpetraVector()),
  Jc(x.getTpetraVector()),
  alpha(alpha_),
  beta(beta_),
  betaType(Scalar),
  diffType(Forward),
  label("NOX::FiniteDifference Jacobian"),
  useGroupForComputeF(false)
{
  // Create the finite difference Jacobian matrix directly using a
  // user supplied graph.
  jacobian = Teuchos::rcp(new TCrsMatrix(Copy, *graph));
  jacobian->FillComplete();
}

FiniteDifference::FiniteDifference(
    Teuchos::ParameterList& printingParams,
    const Teuchos::RCP<NOX::Tpetra::Interface::Required>& i,
    const NOX::Tpetra::Vector& x,
    const Teuchos::RCP<TCrsGraph>& userGraph,
    const Teuchos::RCP<const TVector>& beta_,
    double alpha_) :
  utils(printingParams),
  graph(userGraph),
  interface(i),
  x_perturb(x.getTpetraVector()),
  fo(x.getTpetraVector()),
  fp(x.getTpetraVector()),
  Jc(x.getTpetraVector()),
  alpha(alpha_),
  beta(0),
  betaVector(beta_),
  betaType(Vector),
  diffType(Forward),
  label("NOX::FiniteDifference Jacobian"),
  useGroupForComputeF(false)
{
  // Create the finite difference Jacobian matrix directly using a
  // user supplied graph.
  jacobian = Teuchos::rcp(new TCrsMatrix(Copy, *graph));
  jacobian->FillComplete();
}

FiniteDifference::~FiniteDifference()
{

}

const char* FiniteDifference::Label () const
{
  return label.c_str();
}

int FiniteDifference::SetUseTranspose(bool use_transpose)
{
  return jacobian->SetUseTranspose(use_transpose);
}

int FiniteDifference::Apply(const TMultiVector& X, TMultiVector& Y) const
{
  return jacobian->Apply(X, Y);
}

int FiniteDifference::ApplyInverse(const TMultiVector& X, TMultiVector& Y) const
{
  return jacobian->ApplyInverse(X, Y);
}

bool FiniteDifference::UseTranspose() const
{
  return jacobian->UseTranspose();
}

bool FiniteDifference::HasNormInf() const
{
  return jacobian->HasNormInf();
}

const TMap& FiniteDifference::OperatorDomainMap() const
{
  return jacobian->OperatorDomainMap();
}

const TMap& FiniteDifference::OperatorRangeMap() const
{
  return jacobian->OperatorRangeMap();
}

bool FiniteDifference::Filled() const
{
  return jacobian->Filled();
}

int FiniteDifference::NumMyRowEntries(int MyRow, int & NumEntries) const
{
  return jacobian->NumMyRowEntries(MyRow, NumEntries);
}

int FiniteDifference::MaxNumEntries() const
{
  return jacobian->MaxNumEntries();
}

int FiniteDifference::ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const
{
  return jacobian->ExtractMyRowCopy(MyRow, Length, NumEntries, Values, Indices);
}

int FiniteDifference::ExtractDiagonalCopy(TVector & Diagonal) const
{
  return jacobian->ExtractDiagonalCopy(Diagonal);
}

int FiniteDifference::Multiply(bool TransA, const TMultiVector& X, TMultiVector& Y) const
{
  return jacobian->Multiply(TransA, X, Y);
}

int FiniteDifference::Solve(bool Upper, bool Trans, bool UnitDiagonal, const TMultiVector& X,  TMultiVector& Y) const
{
  return jacobian->Solve(Upper, Trans, UnitDiagonal, X, Y);
}

int FiniteDifference::InvRowSums(TVector& x) const
{
  return jacobian->InvRowSums(x);
}

int FiniteDifference::LeftScale(const TVector& x)
{
  return jacobian->LeftScale(x);
}

int FiniteDifference::InvColSums(TVector& x) const
{
  return jacobian->InvColSums(x);
}

int FiniteDifference::RightScale(const TVector& x)
{
  return jacobian->RightScale(x);
}

double FiniteDifference::NormInf() const
{
  return jacobian->NormInf();
}

double FiniteDifference::NormOne() const
{
  return jacobian->NormOne();
}

#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
int FiniteDifference::NumGlobalNonzeros() const
{
  return jacobian->NumGlobalNonzeros();
}

int FiniteDifference::NumGlobalRows() const
{
  return jacobian->NumGlobalRows();
}

int FiniteDifference::NumGlobalCols() const
{
  return jacobian->NumGlobalCols();
}

int FiniteDifference::NumGlobalDiagonals() const
{
  return jacobian->NumGlobalDiagonals();
}
#endif

long long FiniteDifference::NumGlobalNonzeros64() const
{
  return jacobian->NumGlobalNonzeros64();
}

long long FiniteDifference::NumGlobalRows64() const
{
  return jacobian->NumGlobalRows64();
}

long long FiniteDifference::NumGlobalCols64() const
{
  return jacobian->NumGlobalCols64();
}

long long FiniteDifference::NumGlobalDiagonals64() const
{
  return jacobian->NumGlobalDiagonals64();
}

int FiniteDifference::NumMyNonzeros() const
{
  return jacobian->NumMyNonzeros();
}

int FiniteDifference::NumMyRows() const
{
  return jacobian->NumMyRows();
}

int FiniteDifference::NumMyCols() const
{
  return jacobian->NumMyCols();
}

int FiniteDifference::NumMyDiagonals() const
{
  return jacobian->NumMyDiagonals();
}

bool FiniteDifference::LowerTriangular() const
{
  return jacobian->LowerTriangular();
}

bool FiniteDifference::UpperTriangular() const
{
  return jacobian->UpperTriangular();
}

const Epetra_Comm& FiniteDifference::Comm() const
{
  return jacobian->Comm();
}

const TMap& FiniteDifference::RowMatrixRowMap() const
{
  return jacobian->RowMatrixRowMap();
}

const TMap& FiniteDifference::RowMatrixColMap() const
{
  return jacobian->RowMatrixColMap();
}

const TImport* FiniteDifference::RowMatrixImporter() const
{
  return jacobian->RowMatrixImporter();
}

const TBlockMap& FiniteDifference::Map() const
{
  return jacobian->Map();
}

bool FiniteDifference::computeJacobian(const TVector& x)
{
  return( computeJacobian(x, *this));
}

bool FiniteDifference::computeJacobian(const TVector& x, TOperator& Jac)
{
  // First check to make sure Jac is a NOX::Tpetra::FiniteDifference object
  FiniteDifference* testMatrix = dynamic_cast<FiniteDifference*>(&Jac);
  if (testMatrix == 0) {
    utils.out() << "ERROR: NOX::Tpetra::FiniteDifference::computeJacobian() - "
        << "Jacobian to evaluate is not a FiniteDifference object!"
        << std::endl;
    throw std::runtime_error("NOX Error");
  }

  const TBlockMap& map = fo.Map();

  // We need the TCrsMatrix inside the FiniteDifference object
  // for the correct insertion commands.
  TCrsMatrix& jac = *testMatrix->jacobian;

  // Zero out Jacobian
  jac.PutScalar(0.0);

  // Create an extra perturbed residual vector pointer if needed
  if ( diffType == Centered )
    if ( Teuchos::is_null(fmPtr) )
      fmPtr = Teuchos::rcp(new TVector(x));

  double scaleFactor = 1.0;
  if ( diffType == Backward )
    scaleFactor = -1.0;

  double eta = 0.0;  // Value to perturb the solution vector

#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
  int min = map.MinAllGID();  // Minimum Global ID value
  int max = map.MaxAllGID();  // Maximum Global ID value
  int myMin = map.MinMyGID(); // Minimum Local ID value
  int myMax = map.MaxMyGID(); // Maximum Local ID value
#else
  long long min = map.MinAllGID64();  // Minimum Global ID value
  long long max = map.MaxAllGID64();  // Maximum Global ID value
  long long myMin = map.MinMyGID64(); // Minimum Local ID value
  long long myMax = map.MaxMyGID64(); // Maximum Local ID value
#endif

  // Compute the RHS at the initial solution
  computeF(x, fo, NOX::Tpetra::Interface::Required::FD_Res);

  x_perturb = x;

  // loop over each global unknown
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
int k;
#else
  long long k;
#endif
  for (k = min; k < max+1; k++) {

    // Perturb the solution vector via local IDs only if it is owned by
    // this processor
    int proc = 0;
    if (map.MyGID(k)) {

      if (betaType == Scalar)
    eta = alpha*x[map.LID(k)] + beta;
      else
    eta = alpha*x[map.LID(k)] + (*betaVector)[map.LID(k)];

      x_perturb[map.LID(k)] += scaleFactor * eta;
      proc = map.Comm().MyPID();
    }

    // Find what proc eta is on
    int broadcastProc = 0;
    map.Comm().SumAll(&proc ,&broadcastProc , 1);
    // Send the perturbation variable, eta, to all processors
    map.Comm().Broadcast(&eta, 1, broadcastProc);

    // Compute the perturbed RHS
    computeF(x_perturb,fp, NOX::Tpetra::Interface::Required::FD_Res);

    if ( diffType == Centered ) {
      if (map.MyGID(k))
        x_perturb[map.LID(k)] -= 2.0 * eta;
      computeF(x_perturb,*fmPtr, NOX::Tpetra::Interface::Required::FD_Res);
    }

    // Compute the column k of the Jacobian
    if ( diffType != Centered ) {
      Jc.Update(1.0, fp, -1.0, fo, 0.0);
      Jc.Scale( 1.0/(scaleFactor * eta) );
    }
    else {
      Jc.Update(1.0, fp, -1.0, *fmPtr, 0.0);
      Jc.Scale( 1.0/(2.0 * eta) );
    }

    // Insert nonzero column entries into the jacobian
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
    int j;
#else
    long long j;
#endif
    for (j = myMin; j < myMax+1; j++) { 
      if (!map.MyGID(j))
        continue;
      if (Jc[map.LID(j)] != 0.0) {
        jac.ReplaceGlobalValues(j,1,&Jc[map.LID(j)],&k);
      }
    }

    // Unperturb the solution vector
    x_perturb = x;

  }

  jac.FillComplete();

  return true;
}

bool FiniteDifference::computePreconditioner(const TVector& x,
                         Epetra_Operator& /* Prec */,
                         Teuchos::ParameterList* /* precParams */)
{
  return computeJacobian(x, *this);
}

Teuchos::RCP<TCrsMatrix> FiniteDifference::
createGraphAndJacobian(Interface::Required& /* i */, const TVector& x)
{

  const TBlockMap& map = fo.Map();

  double eta = 0.0;  // Value to perturb the solution vector

#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
  int min = map.MinAllGID();  // Minimum Global ID value
  int max = map.MaxAllGID();  // Maximum Global ID value
  int myMin = map.MinMyGID(); // Minimum Local ID value
  int myMax = map.MaxMyGID(); // Maximum Local ID value
#else
  long long min = map.MinAllGID64();  // Minimum Global ID value
  long long max = map.MaxAllGID64();  // Maximum Global ID value
  long long myMin = map.MinMyGID64(); // Minimum Local ID value
  long long myMax = map.MaxMyGID64(); // Maximum Local ID value
#endif

  // Create the graph
  graph = Teuchos::rcp(new TCrsGraph(Copy,map,10));

  // Compute the RHS at the initial solution
  computeF(x, fo, NOX::Tpetra::Interface::Required::FD_Res);

  // loop over each global unknown
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
  int k;
#else
  long long k;
#endif
  for (k = min; k < max+1; k++) {

    // Perturb the solution vector via local IDs only if it is owned by
    // this processor
    if (map.MyGID(k)) {

      if (betaType == Scalar)
    eta = alpha*x[map.LID(k)] + beta;
      else
    eta = alpha*x[map.LID(k)] + (*betaVector)[map.LID(k)];

      x_perturb[map.LID(k)] += eta;
    }

    // Compute the perturbed RHS
    computeF(x_perturb, fp, NOX::Tpetra::Interface::Required::FD_Res);

    // Compute the column k of the Jacobian
    Jc.Update(1.0, fp, -1.0, fo, 0.0);
    //Jc.Scale(1.0/eta);

    // Insert column entries into the graph
#ifndef TPETRA_NO_32BIT_GLOBAL_INDICES
    int j;
#else
    long long j;
#endif
    for (j = myMin; j < myMax+1; j++) { 
      // Allow for the possibility that rows j from myMin to myMax are not necessarily contigous
      if (!map.MyGID(j))
        continue;
      if (Jc[map.LID(j)] != 0.0) {
        graph->InsertGlobalIndices(j,1,&k);
      }
    }

    // Unperturb the solution vector
    x_perturb = x;

  }

  graph->FillComplete();
  jacobian = Teuchos::rcp(new TCrsMatrix(Copy, *graph));
  jacobian->FillComplete();

  return jacobian;

}

void FiniteDifference::setDifferenceMethod(DifferenceType diffType_)
{
  diffType = diffType_;
}

TCrsMatrix& FiniteDifference::getUnderlyingMatrix() const
{
  return *jacobian;
}

void FiniteDifference::Print(std::ostream& strm) const
{
  jacobian->Print(strm);
}

void FiniteDifference::setGroupForComputeF(NOX::Abstract::Group& group)
{
  useGroupForComputeF = true;
  groupPtr = group.clone();
  return;
}


bool FiniteDifference::computeF(const TVector& input,
                TVector& result,
                NOX::Tpetra::Interface::Required::FillType)
{
  bool ok = false;

  if (!useGroupForComputeF)
    ok = interface->computeF(input, result,
                 NOX::Tpetra::Interface::Required::FD_Res);
  else {

    // Get rid of const for NOX::Tpetra:Vector Ctor.
    TVector& nonconstInput = const_cast<TVector&>(input);

    Teuchos::RCP<TVector> tmpEpetraInputVec =
      //Teuchos::rcp(&nonconstInput, false);
      Teuchos::rcp(new TVector(nonconstInput));
    NOX::Tpetra::Vector noxX(tmpEpetraInputVec,
                 NOX::Tpetra::Vector::CreateCopy);
    groupPtr->setX(noxX);
    groupPtr->computeF();
    result = dynamic_cast<const NOX::Tpetra::Vector&>
      (groupPtr->getF()).getTpetraVector();
    ok = true;

  }

  return ok;
}
