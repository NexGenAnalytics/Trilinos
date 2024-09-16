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

//-----------------------------------------------------------------------------
#ifndef Tpetra_Problem_Interface_HPP
#define Tpetra_Problem_Interface_HPP

// Interface to the NOX_Epetra_Group to provide for
// residual and matrix fill routines.

// ---------- Standard Includes ----------
#include <Tpetra_Vector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>

// ---------- Forward Declarations ----------
template<typename ScalarType>
class Brusselator;

template <typename ScalarType>
class Problem_Interface
{
  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
 
  using tvector_t     = typename Tpetra::Vector<ST, LO, GO>;
  using toperator_t   = typename Tpetra::Operator<ST, LO, GO>;
  using trowmatrix_t  = typename Tpetra::RowMatrix<ST, LO, GO>;

public:
  Problem_Interface(Brusselator<ST>& Problem) : problem(Problem) { }

  //! Compute and return F
  bool computeF(const tvector_t& x,
                tvector_t& FVec)
  {
    return problem.evaluate(&x, &FVec, NULL);
  }

  //! Compute an explicit Jacobian
  bool computeJacobian(const tvector_t& x, toperator_t& Jac)
  {
    return problem.evaluate(&x, 0, 0);
  }
  
  //! Compute the Tpetra_RowMatrix M, that will be used by the Aztec preconditioner instead of the Jacobian.
  /*! This is used when there is no explicit Jacobian present (i.e. Matrix-Free Newton-Krylov).
   * This MUST BE an Epetra_RowMatrix since the Aztec preconditioners need to know the sparsity pattern of the matrix.
   * Returns true if computation was successful.
   */
  bool computePrecMatrix(const tvector_t& x, trowmatrix_t& M)
  {
    std::cout << "ERROR: Problem_Interface::preconditionVector() - Use Explicit Jacobian only for this test problem!" << std::endl;
    throw 1;
  }
  
  //! Computes a user supplied preconditioner based on input vector x. Returns true if computation was successful.
  bool computePreconditioner( const tvector_t& x,
                              toperator_t& Prec,
                              Teuchos::ParameterList* p)
  {
    std::cout << "ERROR: Problem_Interface::preconditionVector() - Use Explicit Jacobian only for this test problem!" << std::endl;
    throw 1;
  }
  
  //! Application Operator: Object that points to the user's evaluation routines.
  /*! This is used to point to the actual routines and to store
   *  auxiliary data required by the user's application for function/Jacobian
   *  evaluations that NOX does not need to know about. This is type of
   *  passdown class design by the application code.
   */
  Brusselator<ST>& problem;
};

#endif
