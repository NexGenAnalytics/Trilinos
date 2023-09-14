// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef TEMPUS_CDR_MODEL_TPETRA_IMPL_HPP
#define TEMPUS_CDR_MODEL_TPETRA_IMPL_HPP

#include "CDR_Model_Functors.hpp"

// Thyra support
#include "Teuchos_Assert.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"
#include "Thyra_VectorStdOps.hpp"

// Tpetra support
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Tpetra_CrsGraph_def.hpp"
#include "Tpetra_CrsMatrix_def.hpp"
#include "Tpetra_Import_def.hpp"
#include "Tpetra_Map_def.hpp"
#include "Tpetra_Vector_def.hpp"
#include <Teuchos_DefaultMpiComm.hpp>

namespace Tempus_Test {

// Constructor

template <typename SC, typename LO, typename GO, typename Node>
CDR_Model_Tpetra<SC, LO, GO, Node>::CDR_Model_Tpetra(
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm,
    const GO num_global_elements, const SC z_min, const SC z_max, const SC a,
    const SC k)
    : comm_(comm), num_global_elements_(num_global_elements), z_min_(z_min),
      z_max_(z_max), a_(a), k_(k), showGetInvalidArg_(false) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using ::Thyra::VectorBase;
  using MEB = ::Thyra::ModelEvaluatorBase;
  using ST = Teuchos::ScalarTraits<SC>;

  TEUCHOS_ASSERT(nonnull(comm_));

  const auto num_nodes = num_global_elements_ + 1;

  // owned space
  x_owned_map_ = rcp(new tpetra_map(num_nodes, 0, comm_));
  x_space_ = ::Thyra::createVectorSpace<SC, LO, GO, Node>(x_owned_map_);

  // ghosted space
  if (comm_->getSize() == 1) {
    x_ghosted_map_ = x_owned_map_;
  } else {

    int OverlapNumMyElements;
    int OverlapgetMinGlobalIndex;
    OverlapNumMyElements = x_owned_map_->getLocalNumElements() + 2;
    if ((comm_->getRank() == 0) || (comm_->getRank() == (comm_->getSize() - 1)))
      OverlapNumMyElements--;

    if (comm_->getRank() == 0)
      OverlapgetMinGlobalIndex = x_owned_map_->getMinGlobalIndex();
    else
      OverlapgetMinGlobalIndex = x_owned_map_->getMinGlobalIndex() - 1;

    Teuchos::Array<GO> overlapMyGlobalNodes(OverlapNumMyElements);

    GO getGlobalElement = OverlapgetMinGlobalIndex;
    for (auto getGlobalElementIter = overlapMyGlobalNodes.begin();
         getGlobalElementIter != overlapMyGlobalNodes.end();
         ++getGlobalElementIter) {
      *getGlobalElementIter = getGlobalElement;
      ++getGlobalElement;
    }

    const auto invalid =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    x_ghosted_map_ =
        Teuchos::rcp(new tpetra_map(invalid, overlapMyGlobalNodes, 0, comm_));
  }

  importer_ = Teuchos::rcp(
      new Tpetra::Import<LO, GO, Node>(x_owned_map_, x_ghosted_map_));

  // residual space
  f_owned_map_ = x_owned_map_;
  f_space_ = x_space_;

  x0_ = ::Thyra::createMember(x_space_);
  V_S(x0_.ptr(), ST::zero());

  // Initialize the graph for W CrsMatrix object
  W_graph_ = createGraph();

  // Create the nodal coordinates
  {
    node_coordinates_ = Teuchos::rcp(new tpetra_vec(x_owned_map_));
    auto length = z_max_ - z_min_;
    auto dx = length/((double) num_global_elements_ - 1);
    auto coordsView = node_coordinates_->getLocalViewHost(Tpetra::Access::ReadWrite);
    for (int i=0; i < x_owned_map_->getLocalNumElements(); i++) {
      coordsView(i, 0) = z_min_ + dx*((double) x_owned_map_->getMinGlobalIndex() + i);
    }

    // node_coordinates_ = Teuchos::rcp(new tpetra_vec(x_owned_map_));
    // SC length = z_max_ - z_min_;
    // const auto mingetGlobalElement = x_owned_map_->getMinGlobalIndex();
    // const auto dz = (z_max_ - z_min_) / static_cast<SC>(num_global_elements_);

    // MeshFillFunctor<tpetra_vec> functor(*node_coordinates_, z_min_, dz,
    //                                     mingetGlobalElement);
    // Kokkos::parallel_for("coords fill", x_owned_map_->getLocalNumElements(),
    //                      functor);
    // Kokkos::fence();
  }

  MEB::InArgsSetup<SC> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_t);
  inArgs.setSupports(MEB::IN_ARG_x);
  inArgs.setSupports(MEB::IN_ARG_beta);
  inArgs.setSupports(MEB::IN_ARG_x_dot);
  inArgs.setSupports(MEB::IN_ARG_alpha);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<SC> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W);
  outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);

  prototypeOutArgs_ = outArgs;

  // Setup nominal values
  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);
  auto x_dot_init = Thyra::createMember(this->get_x_space());
  Thyra::put_scalar(SC(0.0), x_dot_init.ptr());
  nominalValues_.set_x_dot(x_dot_init);

  printf("CDR_Model_Tpetra::CDR_Model_Tpetra\n");
}

// Initializers/Accessors

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Tpetra::CrsGraph<LO, GO, Node>>
CDR_Model_Tpetra<SC, LO, GO, Node>::createGraph()
{
  std::ostream &out = std::cout;
  Teuchos::RCP<Teuchos::FancyOStream> fos =
      Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
  auto W_graph = Teuchos::rcp(new tpetra_graph(x_owned_map_, 5));
  W_graph->describe(*fos,Teuchos::VERB_EXTREME);
  W_graph->resumeFill();

  auto OverlapNumMyElements = static_cast<LO>(x_ghosted_map_->getLocalNumElements());

  // Loop Over # of Finite Elements on Processor
  for (LO elem = 0; elem < OverlapNumMyElements - 1; elem++) {

    // Loop over Nodes in Element
    for (LO i = 0; i < 2; i++) {
      auto row = x_ghosted_map_->getGlobalElement(elem + i);

      // Loop over Trial Functions
      for (LO j = 0; j < 2; j++) {

        // If this row is owned by current processor, add the index
        if (x_owned_map_->isNodeGlobalElement(row)) {
          auto colIndex = x_ghosted_map_->getGlobalElement(elem + j);
          Teuchos::ArrayView<const GO> column(&colIndex, 1);
          W_graph->insertGlobalIndices(row, column);
        }
      }
    }
  }
  W_graph->fillComplete();

  printf("CDR_Model_Tpetra::createGraph\n");
  return W_graph;
}

template <typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::set_x0(
    const Teuchos::ArrayView<const SC> &x0_in) {
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<SC> x0(x0_);
  x0.sv().values()().assign(x0_in);

    printf("CDR_Model_Tpetra::set_x0\n");
}

template <typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::setShowGetInvalidArgs(
    bool showGetInvalidArg) {
  showGetInvalidArg_ = showGetInvalidArg;
      printf("CDR_Model_Tpetra::setShowGetInvalidArgs\n");
}

template <typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::set_W_factory(
    const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<SC>>
        &W_factory) {
  W_factory_ = W_factory;

  printf("CDR_Model_Tpetra::set_W_factory\n");
}

// Public functions overridden from ModelEvaluator

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::VectorSpaceBase<SC>>
CDR_Model_Tpetra<SC, LO, GO, Node>::get_x_space() const {
  printf("CDR_Model_Tpetra::get_x_space\n");
  return x_space_;
}

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::VectorSpaceBase<SC>>
CDR_Model_Tpetra<SC, LO, GO, Node>::get_f_space() const {
  printf("CDR_Model_Tpetra::get_f_space\n");
  return f_space_;
}

template <typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::InArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::getNominalValues() const {
  printf("CDR_Model_Tpetra::getNominalValues\n");
  return nominalValues_;
}

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<Thyra::LinearOpWithSolveBase<double>>
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W() const {
  auto W_factory = this->get_W_factory();

  TEUCHOS_TEST_FOR_EXCEPTION(
      is_null(W_factory), std::runtime_error,
      "W_factory in CDR_Model_Tpetra has a null W_factory!");

  auto matrix = this->create_W_op();
  auto W = Thyra::linearOpWithSolve<SC>(*W_factory, matrix);

  printf("CDR_Model_Tpetra::create_W\n");
  return W;
}

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<Thyra::LinearOpBase<SC>>
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W_op() const {
  auto W_tpetra = Teuchos::rcp(new tpetra_matrix(W_graph_));

  printf("CDR_Model_Tpetra::create_W_op\n");
  return Thyra::tpetraLinearOp<SC, LO, GO, Node>(f_space_, x_space_, W_tpetra);
}

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<::Thyra::PreconditionerBase<SC>>
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W_prec() const {
  auto W_op = create_W_op();
  auto prec = Teuchos::rcp(new Thyra::DefaultPreconditioner<SC>);

  prec->initializeRight(W_op);

  printf("CDR_Model_Tpetra::create_W_prec\n");
  return prec;
}

template <typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<SC>>
CDR_Model_Tpetra<SC, LO, GO, Node>::get_W_factory() const {
  printf("CDR_Model_Tpetra::get_W_factory\n");
  return W_factory_;
}

template <typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::InArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::createInArgs() const {
  printf("CDR_Model_Tpetra::createInArgs\n");
  return prototypeInArgs_;
}

// Private functions overridden from ModelEvaluatorDefaultBase

template <typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::OutArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::createOutArgsImpl() const {
  printf("CDR_Model_Tpetra::createOutArgsImpl\n");
  return prototypeOutArgs_;
}

template <typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<SC> &inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<SC> &outArgs) const {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  std::ostream &out = std::cout;
  RCP<Teuchos::FancyOStream> fos =
      Teuchos::fancyOStream(Teuchos::rcpFromRef(out));

  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));
  TEUCHOS_ASSERT(nonnull(inArgs.get_x_dot()));

  // const Thyra::ConstDetachedVectorView<SC> x(inArgs.get_x());

  auto f_out = outArgs.get_f();
  auto W_out = outArgs.get_W_op();
  auto W_prec_out = outArgs.get_W_prec();

  if (nonnull(f_out) || nonnull(W_out) || nonnull(W_prec_out)) {

    // ****************
    // Get the underlying Tpetra objects
    // ****************

    RCP<tpetra_vec> f;
    if (nonnull(f_out)) {
      f = tpetra_extract::getTpetraVector(outArgs.get_f());
    }

    RCP<tpetra_matrix> J;
    if (nonnull(W_out)) {
      auto W_epetra = tpetra_extract::getTpetraOperator(W_out);
      J = rcp_dynamic_cast<tpetra_matrix>(W_epetra);
      TEUCHOS_ASSERT(nonnull(J));
    }

    RCP<tpetra_matrix> M_inv;
    if (nonnull(W_prec_out)) {
      auto M_tpetra = tpetra_extract::getTpetraOperator(
          W_prec_out->getNonconstRightPrecOp());
      M_inv = rcp_dynamic_cast<tpetra_matrix>(M_tpetra);
      TEUCHOS_ASSERT(nonnull(M_inv));
      J_diagonal_ = Teuchos::rcp(new tpetra_vec(x_owned_map_));
      J_diagonal_->putScalar(0.0);
    }

    // ****************
    // Create ghosted objects
    // ****************

    // Set the boundary condition directly.  Works for both x and xDot solves.
    if (comm_->getRank() == 0) {
      auto x = Teuchos::rcp_const_cast<Thyra::VectorBase<SC>>(inArgs.get_x());
      auto xVec = tpetra_extract::getTpetraVector(x);
      auto xView = xVec->getLocalViewHost(Tpetra::Access::ReadWrite);
      xView(0, 0) = 1.0;
      // JD fix
      // (*tpetra_extract::getConstTpetraVector(x))[0] = 1.0;
    }

    if (is_null(u_ptr))
      u_ptr = Teuchos::rcp(new tpetra_vec(x_ghosted_map_));

    u_ptr->doImport(*(tpetra_extract::getConstTpetraVector(inArgs.get_x())),
                    *importer_, Tpetra::INSERT);

    if (is_null(u_dot_ptr))
      u_dot_ptr = Teuchos::rcp(new tpetra_vec(x_ghosted_map_));

    u_dot_ptr->doImport(
        *(tpetra_extract::getConstTpetraVector(inArgs.get_x_dot())), *importer_,
        Tpetra::INSERT);

    if (is_null(x_ptr)) {
      x_ptr = Teuchos::rcp(new tpetra_vec(x_ghosted_map_));
      x_ptr->doImport(*node_coordinates_, *importer_, Tpetra::INSERT);
    }

    auto u = u_ptr->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto u_dot = u_dot_ptr->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto x = x_ptr->getLocalViewHost(Tpetra::Access::ReadOnly);

    int ierr = 0;
    int OverlapNumMyElements = x_ghosted_map_->getLocalNumElements();

    double xx[2];
    double uu[2];
    double uu_dot[2];
    BasisTpetra basis;
    const auto alpha = inArgs.get_alpha();
    const auto beta = inArgs.get_beta();

    // Zero out the objects that will be filled
    if (nonnull(f))
      f->putScalar(0.0);
    if (nonnull(J))
      J->setAllToScalar(0.0);
    if (nonnull(M_inv))
      M_inv->setAllToScalar(0.0);

    // Loop Over # of Finite Elements on Processor
    for (int ne = 0; ne < OverlapNumMyElements - 1; ne++) {

      // Loop Over Gauss Points
      for (int gp = 0; gp < 2; gp++) {
        // Get the solution and coordinates at the nodes
        xx[0] = x(ne, 0);
        xx[1] = x(ne + 1, 0);
        uu[0] = u(ne, 0);
        uu[1] = u(ne + 1, 0);
        uu_dot[0] = u_dot(ne, 0);
        uu_dot[1] = u_dot(ne + 1, 0);
        // Calculate the basis function at the gauss point
        basis.computeBasis(gp, xx, uu, uu_dot);

        // Loop over Nodes in Element
        for (int i = 0; i < 2; i++) {
          auto row = x_ghosted_map_->getGlobalElement(ne + i);
          // printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
          //      getRank, row, ne+i,x_owned_map_.getGlobalElement(row));
          if (x_owned_map_->getGlobalElement(row)) {
            if (nonnull(f)) {
              {
                auto fView = f->getLocalViewHost(Tpetra::Access::ReadWrite);
                auto val =
                    basis.wt * basis.dz *
                    (basis.uu_dot * basis.phi[i]                 // transient
                     + (a_ / basis.dz * basis.duu * basis.phi[i] // convection
                        + 1.0 / (basis.dz * basis.dz)) *
                           basis.duu * basis.dphide[i]           // diffusion
                     + k_ * basis.uu * basis.uu * basis.phi[i]); // source
                fView(x_owned_map_->getLocalElement(
                          x_ghosted_map_->getGlobalElement(ne + i)),
                      0) += val;
              }

              // (*f)[x_owned_map_->getLocalElement(x_ghosted_map_->getGlobalElement(ne+i))]+=
              //   +basis.wt*basis.dz
              //   *( basis.uu_dot*basis.phi[i] //transient
              //      +(a_/basis.dz*basis.duu*basis.phi[i] // convection
              //      +1.0/(basis.dz*basis.dz))*basis.duu*basis.dphide[i] //
              //      diffusion +k_*basis.uu*basis.uu*basis.phi[i] ); // source
            }
          }
          // Loop over Trial Functions
          if (nonnull(J)) {
            for (int j = 0; j < 2; j++) {
              if (x_owned_map_->getGlobalElement(row)) {
                auto column = x_ghosted_map_->getGlobalElement(ne + j);
                double jac = basis.wt * basis.dz *
                             (alpha * basis.phi[i] * basis.phi[j] // transient
                              + beta * (+a_ / basis.dz * basis.dphide[j] *
                                            basis.phi[i] // convection
                                        + (1.0 / (basis.dz * basis.dz)) *
                                              basis.dphide[j] *
                                              basis.dphide[i] // diffusion
                                        + 2.0 * k_ * basis.uu * basis.phi[j] *
                                              basis.phi[i] // source
                                        ));
                Teuchos::ArrayView<GO> rowIndex(
                    &row, 1); // assuming int as the GlobalOrdinal type
                Teuchos::ArrayView<SC> values(
                    &jac, 1); // assuming double as the Scalar type
                Teuchos::ArrayView<GO> colIndices(
                    &column, 1); // assuming int as the GlobalOrdinal type

                J->sumIntoGlobalValues(rowIndex[0], colIndices, values);
              }
            }
          }
          if (nonnull(M_inv)) {
            M_inv->resumeFill();
            for (int j = 0; j < 2; j++) {
              if (x_owned_map_->getGlobalElement(row)) {
                auto column = x_ghosted_map_->getGlobalElement(ne + j);
                // The prec will be the diagonal of J. No need to assemble the
                // other entries
                if (row == column) {
                  double jac = basis.wt * basis.dz *
                               (alpha * basis.phi[i] * basis.phi[j] // transient
                                + beta * (+a_ / basis.dz * basis.dphide[j] *
                                              basis.phi[i] // convection
                                          + (1.0 / (basis.dz * basis.dz)) *
                                                basis.dphide[j] *
                                                basis.dphide[i] // diffusion
                                          + 2.0 * k_ * basis.uu * basis.phi[j] *
                                                basis.phi[i] // source
                                          ));

                  Teuchos::ArrayView<GO> rowIndex(
                      &row, 1); // assuming int as the GlobalOrdinal type
                  Teuchos::ArrayView<SC> values(
                      &jac, 1); // assuming double as the Scalar type
                  Teuchos::ArrayView<GO> colIndices(
                      &column, 1); // assuming int as the GlobalOrdinal type
                  M_inv->sumIntoGlobalValues(rowIndex[0], colIndices, values);
                }
              }
            }
          }
        }
      }
    }

    // Insert Boundary Conditions and modify Jacobian and function (F)
    // U(0)=1
    if (comm_->getRank() == 0) {
      if (nonnull(f)) {
        auto fView = f->getLocalViewHost(Tpetra::Access::ReadWrite);
        fView(0, 0) = 0.0; // Setting BC above and zero residual here works for
                           // x and xDot solves.
        //(*f)[0]= u[0] - 1.0;   // BC equation works for x solves.
      }
      if (nonnull(J)) {
        J->resumeFill();
        J->replaceGlobalValues(0, Teuchos::tuple<GO>(0, 1),
                               Teuchos::tuple<SC>(1.0, 0.0));
      }
      if (nonnull(M_inv)) {
        M_inv->resumeFill();
        // M_inv->describe(*fos,Teuchos::VERB_EXTREME);
        M_inv->replaceGlobalValues(0, Teuchos::tuple<GO>(0, 1),
                                   Teuchos::tuple<SC>(1.0, 0.0));
        M_inv->fillComplete();
      }
    }

    if (nonnull(J))
      J->fillComplete();

    if (nonnull(M_inv)) {

      // Invert the Jacobian diagonal for the preconditioner
      // For some reason the matrix must be fill complete before calling
      // rightScale


      auto &diag = *J_diagonal_;

      diag.describe(*fos,Teuchos::VERB_EXTREME);
      M_inv->getLocalDiagCopy(diag);

      diag.describe(*fos,Teuchos::VERB_EXTREME);

      diag.reciprocal(diag);
      M_inv->rightScale(diag);
      M_inv->rightScale(diag);


      // typedef Tpetra::CrsMatrix<> crs_matrix_type;
      // typedef typename crs_matrix_type::nonconst_global_inds_host_view_type gids_type;
      // typedef typename crs_matrix_type::nonconst_values_host_view_type vals_type;

      // M_inv->resumeFill();
      // const GO idOfFirstRow = 0;
      // size_t numEntriesInRow = M_inv->getNumEntriesInGlobalRow (idOfFirstRow);
      // vals_type rowvals ("vals",numEntriesInRow);
      // gids_type rowinds ("gids",numEntriesInRow);

      // M_inv->getGlobalRowCopy (idOfFirstRow, rowinds, rowvals, numEntriesInRow);
      // for (size_t i = 0; i < numEntriesInRow; i++) {
      //   if (rowinds[i] == idOfFirstRow) {
      //     // rowvals[i] *= 10.0;
      //     rowvals[i] = 1.0 / rowvals[i];
      //   }
      // }
      // M_inv->replaceGlobalValues (idOfFirstRow, rowinds, rowvals);

      // M_inv->fillComplete();
      M_inv->describe(*fos,Teuchos::VERB_EXTREME);
    }

    TEUCHOS_ASSERT(ierr > -1);
    printf("CDR_Model_Tpetra::evalModelImpl\n");
  }
}

//====================================================================
// Basis vector

// Calculates a linear 1D basis
void BasisTpetra::computeBasis(int gp, double *z, double *u, double *u_dot) {
  constexpr int N = 2;
  if (gp == 0) {
    eta = -1.0 / sqrt(3.0);
    wt = 1.0;
  }
  if (gp == 1) {
    eta = 1.0 / sqrt(3.0);
    wt = 1.0;
  }

  // Calculate basis function and derivatives at nodel pts
  phi[0] = (1.0 - eta) / 2.0;
  phi[1] = (1.0 + eta) / 2.0;
  dphide[0] = -0.5;
  dphide[1] = 0.5;

  // Caculate basis function and derivative at GP.
  dz = 0.5 * (z[1] - z[0]);
  zz = 0.0;
  uu = 0.0;
  duu = 0.0;
  uu_dot = 0.0;
  duu_dot = 0.0;
  for (int i = 0; i < N; i++) {
    zz += z[i] * phi[i];
    uu += u[i] * phi[i];
    duu += u[i] * dphide[i];
    if (u_dot) {
      uu_dot += u_dot[i] * phi[i];
      duu_dot += u_dot[i] * dphide[i];
    }
  }
}

} // namespace Tempus_Test

#endif // TEMPUS_CDR_MODEL_TPETRA_IMPL_HPP
