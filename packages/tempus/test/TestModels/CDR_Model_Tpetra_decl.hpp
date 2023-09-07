// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef TEMPUS_CDR_MODEL_TPETRA_DECL_HPP
#define TEMPUS_CDR_MODEL_TPETRA_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"
#include "Thyra_TpetraThyraWrappers_decl.hpp"

#include <Tpetra_Map_decl.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_Vector_decl.hpp>
#include <Tpetra_Import_decl.hpp>

namespace Tempus_Test {

/** \brief 1D CGFEM model for convection/diffusion/reaction
 *
 * The equation modeled is:

 \verbatim

dT     dT   d^2(T)
-- + a -- + ------ - K * T**2 = 0
dt     dz    dz^2

   subject to:
      T = 1.0 @ z = z_min

 \endverbatim

 * The Matrix <tt>W = d(f)/d(x)</tt> is implemented as a
 * <tt>Thyra::MultiVectorBase</tt> object and the class
 * <tt>Thyra::DefaultSerialDenseLinearOpWithSolveFactory</tt> is used to
 * create the linear solver.
 */
template<typename SC, typename LO, typename GO, typename Node>
class CDR_Model_Tpetra
  : public ::Thyra::StateFuncModelEvaluatorBase<SC>
{
public:
  using tpetra_map = Tpetra::Map<LO, GO, Node>;
  using tpetra_graph = Tpetra::CrsGraph<LO, GO, Node>;
  using tpetra_matrix = Tpetra::CrsMatrix<SC, LO, GO, Node>;
  using tpetra_vec = Tpetra::Vector<SC, LO, GO, Node>;
  using tpetra_extract = ::Thyra::TpetraOperatorVectorExtraction<SC,LO,GO,Node>;

  CDR_Model_Tpetra(const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
            const int num_global_elements,
            const SC z_min,
            const SC z_max,
            const SC a,  // convection
            const SC k); // source

  /** \name Initializers/Accessors */
  //@{

  void set_x0(const Teuchos::ArrayView<const SC> &x0);

  void setShowGetInvalidArgs(bool showGetInvalidArg);

  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<SC> >& W_factory);

  //@}

  /** \name Public functions overridden from ModelEvaluator. */
  //@{

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<SC> > get_x_space() const;
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<SC> > get_f_space() const;
  ::Thyra::ModelEvaluatorBase::InArgs<SC> getNominalValues() const;
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<double> > create_W() const;
  Teuchos::RCP< ::Thyra::LinearOpBase<SC> > create_W_op() const;
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<SC> > get_W_factory() const;
  ::Thyra::ModelEvaluatorBase::InArgs<SC> createInArgs() const;
  Teuchos::RCP< ::Thyra::PreconditionerBase< SC > > create_W_prec() const;
  //@}

private:

  /** Allocates and returns the Jacobian matrix graph */
  virtual Teuchos::RCP<const Tpetra::CrsGraph<LO, GO, Node>> createGraph();

  /** \name Private functions overridden from ModelEvaluatorDefaultBase. */
  //@{

  ::Thyra::ModelEvaluatorBase::OutArgs<SC> createOutArgsImpl() const;
  void evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<SC> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<SC> &outArgs
    ) const;

  //@}

private: // data members

  const Teuchos::RCP<const Teuchos::Comm<int>>  comm_;
  const int num_global_elements_;
  const SC z_min_;
  const SC z_max_;
  const SC a_;
  const SC k_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<SC> > x_space_;
  Teuchos::RCP<const tpetra_map>   x_owned_map_;
  Teuchos::RCP<const tpetra_map>   x_ghosted_map_;
  Teuchos::RCP<const Tpetra::Import<LO, GO, Node>> importer_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<SC> > f_space_;
  Teuchos::RCP<const tpetra_map>   f_owned_map_;

  Teuchos::RCP<tpetra_graph>  W_graph_;

  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<SC> > W_factory_;

  Teuchos::RCP<tpetra_vec> node_coordinates_;
  Teuchos::RCP<tpetra_vec> ghosted_node_coordinates_;

  mutable Teuchos::RCP<tpetra_vec> u_ptr;
  mutable Teuchos::RCP<tpetra_vec> u_dot_ptr;
  mutable Teuchos::RCP<tpetra_vec> x_ptr;

  mutable Teuchos::RCP<tpetra_vec> J_diagonal_;

  ::Thyra::ModelEvaluatorBase::InArgs<SC> nominalValues_;
  Teuchos::RCP< ::Thyra::VectorBase<SC> > x0_;
  Teuchos::Array<SC> p_;
  bool showGetInvalidArg_;
  ::Thyra::ModelEvaluatorBase::InArgs<SC> prototypeInArgs_;
  ::Thyra::ModelEvaluatorBase::OutArgs<SC> prototypeOutArgs_;

};

//==================================================================
// Finite Element Basis Object
class BasisTpetra {

 public:

  // Calculates the values of u and x at the specified gauss point
  void computeBasis(int gp, double *z, double *u, double *u_dot = nullptr);

 public:
  // Variables that are calculated at the gauss point
  std::array<double,2> phi;
  std::array<double,2> dphide;
  double uu = 0.0;
  double zz = 0.0;
  double duu = 0.0;
  double eta = 0.0;
  double wt = 0.0;
  double dz = 0.0;

  // These are only needed for transient
  double uu_dot = 0.0;
  double duu_dot = 0.0;
};

} // namespace Tempus_Test

#endif // TEMPUS_CDR_MODEL_TPETRA_DECL_HPP
