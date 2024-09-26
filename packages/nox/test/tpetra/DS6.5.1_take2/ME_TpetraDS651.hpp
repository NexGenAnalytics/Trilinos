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

#ifndef NOX_TPETRA_ME_DS651_DECL_HPP
#define NOX_TPETRA_ME_DS651_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Teuchos_TimeMonitor.hpp"

template<class Scalar, class LO, class GO, class Node>
class EvaluatorTpetraDS651;

/** \brief Nonmember constuctor.
 *
 * \relates EvaluatorTpetraDS651
 */
template<class Scalar, class LO, class GO, class Node>
Teuchos::RCP<EvaluatorTpetraDS651<Scalar, LO, GO, Node> >
evaluatorTpetraDS651(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                     const Tpetra::global_size_t numGlobalElements,
                     const Scalar zMin,
                     const Scalar zMax);

/** \brief Simple 2 equation test for quadratic and cubic line searches
 * from Dennis & Schnabel's book, Chapter 6.
 *
 * The system modeled is:

  \verbatim

     U0**2 + U1**2 - 2 = 0
     exp(U0-1) + U1**3 -2 = 0

  \endverbatim

  * The Matrix <tt>W = d(f)/d(x)</tt> is implemented as a
  * <tt>Thyra::LinearOpBase</tt> object and the class
  * <tt>Thyra::DefaultSerialDenseLinearOpWithSolveFactory</tt> is used to
  * create the linear solver.
  */


template<class Scalar, class LO, class GO, class Node>
class EvaluatorTpetraDS651 : public ::Thyra::ModelEvaluator<Scalar>
{
public:

  // Public typedefs
  using Scalar = typename Tpetra::Vector<Scalar>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using Node = typename Tpetra::Vector<>::node_type;

  using tpetra_map    = Tpetra::Map<LO, GO, Node>;
  using tpetra_vec    = Tpetra::Vector<Scalar, LO, GO, Node>;
  using tpetra_graph  = Tpetra::CrsGraph<LO, GO, Node>;
  using tpetra_matrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;

  using thyra_op        = ::Thyra::LinearOpBase<Scalar>;
  using thyra_vec       = ::Thyra::VectorBase<Scalar>;
  using thyra_mvec      = ::Thyra::MultiVectorBase<Scalar>;
  using thyra_prec      = ::Thyra::PreconditionerBase<Scalar>;
  using thyra_vec_space = ::Thyra::VectorSpaceBase<Scalar>;

  // Constructor
  EvaluatorTpetraDS651(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                       const Tpetra::global_size_t numGlobalElements,
                       const Scalar zMin,
                       const Scalar zMax);

  /** \name Initializers/Accessors */
  //@{

  /** \brief . */
  void set_x0(const Teuchos::ArrayView<const Scalar> &x0);

  /** \brief . */
  void setShowGetInvalidArgs(bool showGetInvalidArg);

  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);

  //@}

  /** \name Public functions overridden from ModelEvaulator. */
  //@{

  Teuchos::RCP<const thyra_vec_space> get_x_space() const override;
  Teuchos::RCP<const thyra_vec_space> get_f_space() const override;
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const override;
  Teuchos::RCP<thyra_op> create_W_op() const override;
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const override;
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const override;
  Teuchos::RCP<thyra_prec> create_W_prec() const override;

  // These are for constraint solver support
  int Np () const;
  int Ng () const;
  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgs() const override;

}

//==================================================================
#include "ME_TpetraDS651_def.hpp"
//==================================================================

#endif