// $Id$
// $Source$

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

#include "NOX_Tpetra_Vector.hpp"
#include "NOX_Tpetra_MultiVector.hpp"
#include "TVector.h"
#include "NOX_Tpetra_VectorSpace_L2.H"

NOX::Tpetra::Vector::
Vector(const Teuchos::RCP<TVector>& source,
       NOX::Tpetra::Vector::MemoryType memoryType,
       NOX::CopyType type,
       Teuchos::RCP<NOX::Tpetra::VectorSpace> vs)
{
  if (Teuchos::is_null(vs))
    vectorSpace = Teuchos::rcp(new NOX::Tpetra::VectorSpaceL2);
  else
    vectorSpace = vs;

  if (memoryType == NOX::Tpetra::Vector::CreateView)
    tpetraVec = source;
  else {

    switch (type) {

    case DeepCopy:        // default behavior

      tpetraVec = Teuchos::rcp(new TVector(*source));
      break;

    case ShapeCopy:

      tpetraVec = Teuchos::rcp(new TVector(source->Map()));
      break;
    }

  }
}

NOX::Tpetra::Vector::Vector(const TVector& source, NOX::CopyType type,
                Teuchos::RCP<NOX::Tpetra::VectorSpace> vs)
{
  if (Teuchos::is_null(vs))
    vectorSpace = Teuchos::rcp(new NOX::Tpetra::VectorSpaceL2);
  else
    vectorSpace = vs;

  switch (type) {

  case DeepCopy:        // default behavior

    tpetraVec = Teuchos::rcp(new TVector(source));
    break;

  case ShapeCopy:

    tpetraVec = Teuchos::rcp(new TVector(source.Map()));
    break;

  }
}

NOX::Tpetra::Vector::Vector(const NOX::Tpetra::Vector& source,
                NOX::CopyType type)
{
  vectorSpace = source.vectorSpace;

  switch (type) {

  case DeepCopy:        // default behavior

    tpetraVec = Teuchos::rcp(new TVector(source.getTpetraVector()));
    break;

  case ShapeCopy:

    epetraVec =
      Teuchos::rcp(new TVector(source.getTpetraVector().Map()));
    break;

  }
}

NOX::Tpetra::Vector::~Vector()
{

}

NOX::Abstract::Vector& NOX::Tpetra::Vector::operator=(const TVector& source)
{
  tpetraVec->Scale(1.0, source);
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::operator=(const NOX::Abstract::Vector& source)
{
  return operator=(dynamic_cast<const NOX::Tpetra::Vector&>(source));
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::operator=(const NOX::Tpetra::Vector& source)
{
  tpetraVec->Scale(1.0, source.getTpetraVector());
  return *this;
}

TVector& NOX::Tpetra::Vector::getTpetraVector()
{
  return *tpetraVec;
}

const TVector& NOX::Tpetra::Vector::getTpetraVector() const
{
  return *tpetraVec;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::init(double value)
{
  tpetraVec->PutScalar(value);
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::random(bool useSeed, int seed)
{
  if (useSeed)
    tpetraVec->SetSeed(seed);
  tpetraVec->Random();
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::abs(const NOX::Abstract::Vector& base)
{
  return abs(dynamic_cast<const NOX::Tpetra::Vector&>(base));
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::abs(const NOX::Tpetra::Vector& base)
{
  tpetraVec->Abs(base.getTpetraVector());
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::reciprocal(const NOX::Abstract::Vector& base)
{
  return reciprocal(dynamic_cast<const NOX::Tpetra::Vector&>(base));
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::reciprocal(const NOX::Tpetra::Vector& base)
{
  tpetraVec->Reciprocal(base.getTpetraVector());
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::scale(double alpha)
{
  tpetraVec->Scale(alpha);
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::
update(double alpha, const NOX::Abstract::Vector& a, double gamma)
{
  return update(alpha, dynamic_cast<const NOX::Tpetra::Vector&>(a), gamma);
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::
update(double alpha, const NOX::Tpetra::Vector& a, double gamma)
{
  tpetraVec->Update(alpha, a.getTpetraVector(), gamma);
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::
update(double alpha, const NOX::Abstract::Vector& a,
       double beta, const NOX::Abstract::Vector& b,
       double gamma)
{
  return update(alpha, dynamic_cast<const NOX::Tpetra::Vector&>(a),
        beta, dynamic_cast<const NOX::Tpetra::Vector&>(b), gamma);
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::
update(double alpha, const NOX::Tpetra::Vector& a,
       double beta, const NOX::Tpetra::Vector& b,
       double gamma)
{
  tpetraVec->Update(alpha, a.getTpetraVector(), beta, b.getTpetraVector(), gamma);
  return *this;
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::scale(const NOX::Abstract::Vector& a)
{
  return scale(dynamic_cast<const NOX::Tpetra::Vector&>(a));
}

NOX::Abstract::Vector& NOX::Tpetra::Vector::scale(const NOX::Tpetra::Vector& a)
{
  tpetraVec->multiply(1.0, *tpetraVec, a.getTpetraVector(), 0.0);
  return *this;
}

Teuchos::RCP<NOX::Abstract::Vector> NOX::Tpetra::Vector::
clone(CopyType type) const
{
  Teuchos::RCP<NOX::Abstract::Vector> newVec =
    Teuchos::rcp(new NOX::Tpetra::Vector(*tpetraVec, type, vectorSpace));
  return newVec;
}

Teuchos::RCP<NOX::Abstract::MultiVector>
NOX::Tpetra::Vector::createMultiVector(
                    const NOX::Abstract::Vector* const* vecs,
                    int numVecs, NOX::CopyType type) const
{
  if (numVecs < 0) {
    std::cerr << "NOX::Tpetra::Vector::createMultiVector:  Error!  Multivector"
     << " must have positive number of columns!" << std::endl;
    throw std::runtime_error("NOX Error");
  }

  double** v = new double*[numVecs+1];
  const TBlockMap& map = tpetraVec->Map();
  const TVector* vecPtr;

  tpetraVec->ExtractView(&(v[0]));
  for (int i=0; i<numVecs; i++) {
    const NOX::Tpetra::Vector & noxEpetraVecPtr =
      dynamic_cast<const NOX::Tpetra::Vector&>(*vecs[i]);
    vecPtr = &(noxEpetraVecPtr.getTpetraVector());
    vecPtr->ExtractView(&(v[i+1]));
  }

  TMultiVector tpetra_mv(View, map, v, numVecs+1);

  Teuchos::RCP<NOX::Tpetra::MultiVector> mv =
    Teuchos::rcp(new NOX::Tpetra::MultiVector(tpetra_mv, type));

  delete [] v;

  return mv;
}

Teuchos::RCP<NOX::Abstract::MultiVector>
NOX::Tpetra::Vector::createMultiVector(int numVecs, NOX::CopyType type) const
{
  if (numVecs <= 0) {
    std::cerr << "NOX::Tpetra::Vector::createMultiVector:  Error!  Multivector"
     << " must have positive number of columns!" << std::endl;
    throw std::runtime_error("NOX Error");
  }

  const TBlockMap& map = tpetraVec->Map();
  TMultiVector *tpetra_mv;

  if (type == NOX::ShapeCopy)
   tpetra_mv = new TMultiVector(map, numVecs, true);
  else {
   tpetra_mv = new TMultiVector(map, numVecs, false);
    TVector* v;
    for (int i=0; i<numVecs; i++) {
      v = (*tpetra_mv)(i);
      *v = *tpetraVec;
    }
  }

  Teuchos::RCP<NOX::Tpetra::MultiVector> mv =
    Teuchos::rcp(new NOX::Tpetra::MultiVector(*tpetra_mv, type));

  delete tpetra_mv;

  return mv;
}

double NOX::Tpetra::Vector::norm(NOX::Abstract::Vector::NormType type) const
{
  return vectorSpace->norm(*tpetraVec, type);
}

double NOX::Tpetra::Vector::norm(const NOX::Abstract::Vector& weights) const
{
  return norm(dynamic_cast<const NOX::Tpetra::Vector&>(weights));
}

double NOX::Tpetra::Vector::norm(const NOX::Tpetra::Vector& /* weights */) const
{
    std::cerr << "NOX::Tpetra::Vector - Weighted norm not supported" << std::endl;
    throw std::runtime_error("NOX-Epetra Error");
}

double NOX::Tpetra::Vector::innerProduct(const NOX::Abstract::Vector& y) const
{
  return innerProduct(dynamic_cast<const NOX::Tpetra::Vector&>(y));
}

double NOX::Tpetra::Vector::innerProduct(const NOX::Tpetra::Vector& y) const
{
  return vectorSpace->innerProduct(*tpetraVec, y.getTpetraVector());
}

NOX::size_type NOX::Tpetra::Vector::length() const
{
  return tpetraVec->GlobalLength64();
}

void NOX::Tpetra::Vector::print(std::ostream& stream) const
{
  tpetraVec->Print(stream);
  return;
}

Teuchos::RCP<NOX::Tpetra::VectorSpace>
NOX::Tpetra::Vector::getVectorSpace() const
{
  return vectorSpace;
}
