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
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"

// Tpetra support
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include <Teuchos_DefaultMpiComm.hpp>
#include "Tpetra_Map_def.hpp"
#include "Tpetra_Vector_def.hpp"
#include "Tpetra_Import_def.hpp"
#include "Tpetra_CrsGraph_def.hpp"
#include "Tpetra_CrsMatrix_def.hpp"

namespace Tempus_Test {

// Constructor

template<typename SC, typename LO, typename GO, typename Node>
CDR_Model_Tpetra<SC, LO, GO, Node>::
CDR_Model_Tpetra(const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
          const int num_global_elements,
          const SC z_min,
          const SC z_max,
          const SC a,
          const SC k) :
  comm_(comm),
  num_global_elements_(num_global_elements),
  z_min_(z_min),
  z_max_(z_max),
  a_(a),
  k_(k),
  showGetInvalidArg_(false)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using ::Thyra::VectorBase;
  typedef ::Thyra::ModelEvaluatorBase MEB;
  typedef Teuchos::ScalarTraits<SC> ST;

  TEUCHOS_ASSERT(nonnull(comm_));

  const int num_nodes = num_global_elements_ + 1;

  // owned space
  x_owned_map_ = rcp(new tpetra_map(num_nodes, 0, comm_));
  x_space_ = ::Thyra::create_VectorSpace(x_owned_map_);

  // ghosted space
  if (comm_->getSize() == 1) {
    x_ghosted_map_ = x_owned_map_;
  } else {

    int OverlapNumMyElements;
    int OverlapgetMinGlobalIndex;
    OverlapNumMyElements = x_owned_map_->getLocalNumElements() + 2;
    if ( (comm_->getRank() == 0) || (comm_->getRank() == (comm_->getSize() - 1)) )
      OverlapNumMyElements --;

    if (comm_->getRank() == 0)
      OverlapgetMinGlobalIndex = x_owned_map_->getMinGlobalIndex();
    else
      OverlapgetMinGlobalIndex = x_owned_map_->getMinGlobalIndex() - 1;

    Teuchos::Array<GO> overlapMyGlobalNodes(OverlapNumMyElements);

    GO gid = OverlapgetMinGlobalIndex;
    for (auto gidIter = overlapMyGlobalNodes.begin(); gidIter != overlapMyGlobalNodes.end(); ++gidIter) {
      *gidIter = gid;
      ++gid;
    }

    const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    x_ghosted_map_ =
      Teuchos::rcp(new tpetra_map(invalid, overlapMyGlobalNodes, 0, comm_));
  }

  importer_ = Teuchos::rcp(new Tpetra::Import<LO, GO, Node>(x_owned_map_, x_ghosted_map_));
//   set_x0(Teuchos::tuple<SC>(x0, x1)());

  // Initialize the graph for W CrsMatrix object
  W_graph_ = createGraph();

  // Create the nodal coordinates
  {
    node_coordinates_ = Teuchos::rcp(new tpetra_vec(x_owned_map_));
    SC length = z_max_ - z_min_;
    const auto minGID = x_owned_map_->getMinGlobalIndex();
    const auto dz = (z_max_ - z_min_)/static_cast<SC>(num_global_elements_);

    MeshFillFunctor<tpetra_vec> func(*node_coordinates_, z_min_, dz, minGID);
    // for (int i=0; i < x_owned_map_->getLocalNumElements(); i++) {
    //   (*node_coordinates_)[i] = z_min_ + dx*((double) x_owned_map_->getMinGlobalIndex() + i);
    // }
  }



  MEB::InArgsSetup<SC> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports( MEB::IN_ARG_t );
  inArgs.setSupports( MEB::IN_ARG_x );
  inArgs.setSupports( MEB::IN_ARG_beta );
  inArgs.setSupports( MEB::IN_ARG_x_dot );
  inArgs.setSupports( MEB::IN_ARG_alpha );
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<SC> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W);
  outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);
//   outArgs.set_W_properties(DerivativeProperties(
//                  DERIV_LINEARITY_NONCONST
//                  ,DERIV_RANK_FULL
//                  ,true // supportsAdjoint
//                  ));
  prototypeOutArgs_ = outArgs;

  // Setup nominal values
  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);
  auto x_dot_init = Thyra::createMember(this->get_x_space());
  Thyra::put_scalar(SC(0.0),x_dot_init.ptr());
  nominalValues_.set_x_dot(x_dot_init);
}

// Initializers/Accessors

template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Tpetra::CrsGraph<LO, GO, Node>>
CDR_Model_Tpetra<SC, LO, GO, Node>::createGraph()
{
  using size_type = typename tpetra_graph::local_graph_device_type::size_type;

  // Compute graph offset array
  int numProcs = comm_->getSize();
  int myRank = comm_->getRank();
  std::size_t numMyNodes = x_owned_map_->getLocalNumElements();
  std::size_t numLocalEntries = 0;
  //Kokkos::View<std::size_t*> counts("row counts", numMyNodes);
  Kokkos::View<size_type*> counts("row counts", numMyNodes);
  {
    RowCountsFunctor<size_type, LO> functor(counts, numMyNodes, numProcs, myRank);
    Kokkos::parallel_reduce("row counts comp", numMyNodes, functor, numLocalEntries);
  }

  //Kokkos::View<std::size_t*> offsets("row offsets", numMyNodes+1);
  Kokkos::View<size_type*> offsets("row offsets", numMyNodes+1);
  {
    RowOffsetsFunctor<size_type, LO> functor(offsets, counts, numMyNodes);
    Kokkos::parallel_scan("row offsets comp", numMyNodes+1, functor);
  }

  // Create array of non-zero entry column indices
  Kokkos::View<LO*> indices("column indices", numLocalEntries);
  //typename local_graph_type::entries_type::non_const_type indices("column indices", numLocalEntries);
  {
    ColumnIndexCompFunctor<size_type, LO> functor(indices, offsets, counts, numMyNodes, numProcs, myRank);
    Kokkos::parallel_for("column indices comp", numMyNodes, functor);
  }

  //Sort the indices within each row.
  Tpetra::Import_Util::sortCrsEntries(offsets, indices);

  // Construct the graph
  Teuchos::RCP<tpetra_graph> W_graph =
    Teuchos::rcp(new tpetra_graph(x_owned_map_, x_ghosted_map_, offsets, indices));
  W_graph->fillComplete();
  return W_graph;
}

template<typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::set_x0(const Teuchos::ArrayView<const SC> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<SC> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<SC> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaluator


template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::VectorSpaceBase<SC> >
CDR_Model_Tpetra<SC, LO, GO, Node>::get_x_space() const
{
  return x_space_;
}


template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::VectorSpaceBase<SC> >
CDR_Model_Tpetra<SC, LO, GO, Node>::get_f_space() const
{
  return f_space_;
}


template<typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::InArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::getNominalValues() const
{
  return nominalValues_;
}


template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<Thyra::LinearOpWithSolveBase<double> >
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W() const
{
  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<double> > W_factory =
    this->get_W_factory();

  TEUCHOS_TEST_FOR_EXCEPTION(is_null(W_factory),std::runtime_error,"W_factory in CDR_Model_Tpetra has a null W_factory!");

  Teuchos::RCP<Thyra::LinearOpBase<double> > matrix = this->create_W_op();

  Teuchos::RCP<Thyra::LinearOpWithSolveBase<double> > W =
    Thyra::linearOpWithSolve<double>(*W_factory,matrix);

  return W;
}

template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<Thyra::LinearOpBase<SC> >
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W_op() const
{
  auto W_tpetra = Teuchos::rcp(new tpetra_matrix(W_graph_));

  return Thyra::tpetraLinearOp<SC, LO, GO, Node>(f_space_, x_space_, W_tpetra);
}

template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP< ::Thyra::PreconditionerBase<SC> >
CDR_Model_Tpetra<SC, LO, GO, Node>::create_W_prec() const
{
  auto W_op = create_W_op();
  auto prec = Teuchos::rcp(new Thyra::DefaultPreconditioner<SC>);

  prec->initializeRight(W_op);

  return prec;
}

template<typename SC, typename LO, typename GO, typename Node>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<SC> >
CDR_Model_Tpetra<SC, LO, GO, Node>::get_W_factory() const
{
  return W_factory_;
}


template<typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::InArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaluatorDefaultBase


template<typename SC, typename LO, typename GO, typename Node>
Thyra::ModelEvaluatorBase::OutArgs<SC>
CDR_Model_Tpetra<SC, LO, GO, Node>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<typename SC, typename LO, typename GO, typename Node>
void CDR_Model_Tpetra<SC, LO, GO, Node>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<SC> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<SC> &outArgs
  ) const
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));
  TEUCHOS_ASSERT(nonnull(inArgs.get_x_dot()));

  //const Thyra::ConstDetachedVectorView<SC> x(inArgs.get_x());

  auto f_out = outArgs.get_f();
  auto W_out = outArgs.get_W_op();
  auto W_prec_out = outArgs.get_W_prec();


  if ( nonnull(f_out) || nonnull(W_out) || nonnull(W_prec_out) ) {

    // ****************
    // Get the underlying epetra objects
    // ****************

    RCP<tpetra_vec> f;
    if (nonnull(f_out)) {
      f = tpetra_extract::getConstTpetraVector(outArgs.get_f());
    }

    RCP<Tpetra::CrsMatrix<SC, LO, GO, Node>> J;
    if (nonnull(W_out)) {
      auto W_epetra = tpetra_extract::getTpetraOperator(W_out);
      J = rcp_dynamic_cast<tpetra_matrix>(W_epetra);
      TEUCHOS_ASSERT(nonnull(J));
    }

    RCP<Tpetra::CrsMatrix<SC, LO, GO, Node>> M_inv;
    if (nonnull(W_prec_out)) {
      auto M_epetra = tpetra_extract::getTpetraOperator(W_prec_out->getNonconstRightPrecOp());
      M_inv = rcp_dynamic_cast<tpetra_matrix>(M_epetra);
      TEUCHOS_ASSERT(nonnull(M_inv));
      J_diagonal_ = Teuchos::rcp(new Tpetra::Vector(x_owned_map_));
    }

    // ****************
    // Create ghosted objects
    // ****************

    // Set the boundary condition directly.  Works for both x and xDot solves.
    if (comm_->getRank() == 0) {
      RCP<Thyra::VectorBase<SC> > x = Teuchos::rcp_const_cast<Thyra::VectorBase<SC> > (inArgs.get_x());
      // JD fix
      // (*tpetra_extract::getConstTpetraVector(x))[0] = 1.0;
    }

    if (is_null(u_ptr))
      u_ptr = Teuchos::rcp(new Tpetra::Vector(x_ghosted_map_));

    u_ptr->doImport(*(tpetra_extract::getConstTpetraVector(inArgs.get_x())), *importer_, Tpetra::INSERT);

    if (is_null(u_dot_ptr))
      u_dot_ptr = Teuchos::rcp(new Tpetra::Vector(x_ghosted_map_));

    u_dot_ptr->doImport(*(tpetra_extract::getConstTpetraVector(inArgs.get_x_dot())), *importer_, Tpetra::INSERT);

    if (is_null(x_ptr)) {
      x_ptr = Teuchos::rcp(new Tpetra::Vector(x_ghosted_map_));
      x_ptr->doImport(*node_coordinates_, *importer_, Tpetra::INSERT);
    }

    auto& u = *u_ptr;
    auto& u_dot = *u_dot_ptr;
    auto& x = *x_ptr;

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
    // for (int ne=0; ne < OverlapNumMyElements-1; ne++) {

    //   // Loop Over Gauss Points
    //   for(int gp=0; gp < 2; gp++) {
    //     // Get the solution and coordinates at the nodes
    //     xx[0]=x[ne];
    //     xx[1]=x[ne+1];
    //     uu[0]=u[ne];
    //     uu[1]=u[ne+1];
    //     uu_dot[0]=u_dot[ne];
    //     uu_dot[1]=u_dot[ne+1];
    //     // Calculate the basis function at the gauss point
    //     basis.computeBasis(gp, xx, uu, uu_dot);

    //     // Loop over Nodes in Element
    //     for (int i=0; i< 2; i++) {
    //       int row=x_ghosted_map_->GID(ne+i);
    //       //printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
    //       //     getRank, row, ne+i,x_owned_map_.MyGID(row));
    //       if (x_owned_map_->MyGID(row)) {
    //         if (nonnull(f)) {
    //           (*f)[x_owned_map_->LID(x_ghosted_map_->GID(ne+i))]+=
    //             +basis.wt*basis.dz
    //             *( basis.uu_dot*basis.phi[i] //transient
    //                +(a_/basis.dz*basis.duu*basis.phi[i] // convection
    //                +1.0/(basis.dz*basis.dz))*basis.duu*basis.dphide[i] // diffusion
    //                +k_*basis.uu*basis.uu*basis.phi[i] ); // source
    //         }
    //       }
    //       // Loop over Trial Functions
    //       if (nonnull(J)) {
    //         for(int j=0;j < 2; j++) {
    //           if (x_owned_map_->MyGID(row)) {
    //             int column=x_ghosted_map_->GID(ne+j);
    //             double jac=
    //               basis.wt*basis.dz*(
    //                                  alpha * basis.phi[i] * basis.phi[j] // transient
    //                                  + beta * (
    //                                            +a_/basis.dz*basis.dphide[j]*basis.phi[i] // convection
    //                                            +(1.0/(basis.dz*basis.dz))*basis.dphide[j]*basis.dphide[i] // diffusion
    //                                            +2.0*k_*basis.uu*basis.phi[j]*basis.phi[i] // source
    //                                            )
    //                                  );
    //             ierr=J->SumIntoGlobalValues(row, 1, &jac, &column);
    //           }
    //         }
    //       }
    //       if (nonnull(M_inv)) {
    //         for(int j=0;j < 2; j++) {
    //           if (x_owned_map_->MyGID(row)) {
    //             int column=x_ghosted_map_->GID(ne+j);
    //             // The prec will be the diagonal of J. No need to assemble the other entries
    //             if (row == column) {
    //               double jac=
    //                 basis.wt*basis.dz*(
    //                                    alpha * basis.phi[i] * basis.phi[j] // transient
    //                                    + beta * (
    //                                              +a_/basis.dz*basis.dphide[j]*basis.phi[i] // convection
    //                                              +(1.0/(basis.dz*basis.dz))*basis.dphide[j]*basis.dphide[i] // diffusion
    //                                              +2.0*k_*basis.uu*basis.phi[j]*basis.phi[i] // source
    //                                              )
    //                                    );
    //               ierr = M_inv->SumIntoGlobalValues(row, 1, &jac, &column);
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    // Insert Boundary Conditions and modify Jacobian and function (F)
    // U(0)=1
    if (comm_->getRank() == 0) {
      if (nonnull(f)){
        auto fView = f->getLocalViewHost(Tpetra::Access::ReadWrite);
        fView(0,0) = 0.0;           // Setting BC above and zero residual here works for x and xDot solves.
        //(*f)[0]= u[0] - 1.0;   // BC equation works for x solves.
      } if (nonnull(J)) {
        int column=0;
        double jac=1.0;
        J->insertGlobalValues(0, Teuchos::tuple<GO>(column), Teuchos::tuple<SC>(jac));
        column=1;
        jac=0.0;
        J->insertGlobalValues(0, Teuchos::tuple<GO>(column), Teuchos::tuple<SC>(jac));
      }
      if (nonnull(M_inv)) {
        int column=0;
        double jac=1.0;
        M_inv->insertGlobalValues(0, Teuchos::tuple<GO>(column), Teuchos::tuple<SC>(jac));
        column=1;
        jac=0.0;
        M_inv->insertGlobalValues(0, Teuchos::tuple<GO>(column), Teuchos::tuple<SC>(jac));
      }
    }

    if (nonnull(J))
      J->fillComplete();

    if (nonnull(M_inv)) {
      // Invert the Jacobian diagonal for the preconditioner
      // For some reason the matrix must be fill complete before calling rightScale
      M_inv->fillComplete();
      auto& diag = *J_diagonal_;
      M_inv->getLocalDiagCopy(diag);
      diag.reciprocal(diag);
      M_inv->rightScale(diag);
      M_inv->rightScale(diag);
    }

    TEUCHOS_ASSERT(ierr > -1);

  }

}

//====================================================================
// Basis vector

// Calculates a linear 1D basis
void BasisTpetra::computeBasis(int gp, double *z, double *u, double *u_dot) {
  constexpr int N = 2;
  if (gp==0) {eta=-1.0/sqrt(3.0); wt=1.0;}
  if (gp==1) {eta=1.0/sqrt(3.0); wt=1.0;}

  // Calculate basis function and derivatives at nodel pts
  phi[0]=(1.0-eta)/2.0;
  phi[1]=(1.0+eta)/2.0;
  dphide[0]=-0.5;
  dphide[1]=0.5;

  // Caculate basis function and derivative at GP.
  dz=0.5*(z[1]-z[0]);
  zz=0.0;
  uu=0.0;
  duu=0.0;
  uu_dot=0.0;
  duu_dot=0.0;
  for (int i=0; i < N; i++) {
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
