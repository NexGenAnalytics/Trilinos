#ifndef NOX_TPETRA_ME_DS651_DEF_HPP
#define NOX_TPETRA_ME_DS651_DEF_HPP

// Thyra support
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"
#include "Tpetra_Import_Util2.hpp"  //for sortCrsEntries

// Tpetra support
#include "Thyra_TpetraThyraWrappers.hpp"
#include "NOX_TpetraTypedefs.hpp"

// Kokkos support
#include "Kokkos_Core.hpp"

#include "DS651_Functors.hpp"

// Nonmember constuctors

template<class Scalar, class LO, class GO, class Node>
Teuchos::RCP<EvaluatorTpetraDS651<Scalar, LO, GO, Node> >
evaluatorTpetraDS651(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                     const Tpetra::global_size_t numGlobalElements,
                     const Scalar zMin,
                     const Scalar zMax)
{
  return Teuchos::rcp(new EvaluatorTpetraDS651<Scalar, LO, GO, Node>(comm,numGlobalElements,zMin,zMax));
}

// Constructor

template<class Scalar, class LO, class GO, class Node>
EvaluatorTpetraDS651<Scalar, LO, GO, Node>::
EvaluatorTpetraDS651(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                     const Tpetra::global_size_t numGlobalElements,
                     const Scalar zMin,
                     const Scalar zMax) :
  comm_(comm),
  numGlobalElements_(numGlobalElements),
  zMin_(zMin),
  zMax_(zMax),
  Np_(5),
  Ng_(7),
  printDebug_(false),
  showGetInvalidArg_(false),
  pNames_(Np_),
  gNames_(Ng_)
{
  typedef ::Thyra::ModelEvaluatorBase MEB;
  typedef Teuchos::ScalarTraits<scalar_type> ST;

  TEUCHOS_ASSERT(nonnull(comm_));

  const Tpetra::global_size_t numNodes = numGlobalElements_ + 1;

  // owned space
  GO indexBase = 0;
  xOwnedMap_ = Teuchos::rcp(new const tpetra_map(numNodes, indexBase, comm_));
  xSpace_ = ::Thyra::createVectorSpace<Scalar, LO, GO, Node>(xOwnedMap_);

  // ghosted space
  if (comm_->getSize() == 1) {
    xGhostedMap_ = xOwnedMap_;
  } else {
    std::size_t overlapNumMyNodes;
    GO overlapMinMyGID;
    overlapNumMyNodes = xOwnedMap_->getLocalNumElements() + 2;
    if ( (comm_->getRank() == 0) || (comm_->getRank() == (comm_->getSize() - 1)) )
      --overlapNumMyNodes;

    if (comm_->getRank() == 0)
      overlapMinMyGID = xOwnedMap_->getMinGlobalIndex();

    else
      overlapMinMyGID = xOwnedMap_->getMinGlobalIndex() - 1;

    Teuchos::Array<GO> overlapMyGlobalNodes(overlapNumMyNodes);
    GO gid = overlapMinMyGID;
    for (auto gidIter = overlapMyGlobalNodes.begin(); gidIter != overlapMyGlobalNodes.end(); ++gidIter) {
      *gidIter = gid;
      ++gid;
    }

    const Tpetra::global_size_t invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    xGhostedMap_ = Teuchos::rcp(new tpetra_map(invalid, overlapMyGlobalNodes, indexBase, comm_));
  }

  importer_ = Teuchos::rcp(new Tpetra::Import<LO,GO,Node>(xOwnedMap_, xGhostedMap_));

  // residual space
  fOwnedMap_ = xOwnedMap_;
  fSpace_ = xSpace_;

  x0_ = ::Thyra::createMember(xSpace_);
  V_S(x0_.ptr(), ST::zero());

  // Initialize the graph for W CrsMatrix object
  W_graph_ = createGraph();

  // Create the nodal coorinates
  std::size_t numLocalNodes = xOwnedMap_->getLocalNumElements();
  GO minGID = xOwnedMap_->getMinGlobalIndex();
  Scalar dz = (zMax_ - zMin_)/static_cast<Scalar>(numGlobalElements_);
  nodeCoordinates_ = Teuchos::rcp(new tpetra_vec(xOwnedMap_));

  MeshFillFunctor<tpetra_vec> functor(*nodeCoordinates_, zMin_, dz, minGID);
  Kokkos::parallel_for("coords fill", numLocalNodes, functor);

  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_x);
  inArgs.set_Np_Ng(Np_,Ng_);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);
  outArgs.set_Np_Ng(Np_,Ng_);
  outArgs.setSupports(MEB::OUT_ARG_DfDp,2,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDx,4,MEB::DerivativeSupport(MEB::DERIV_MV_GRADIENT_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDp,4,2,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));

  outArgs.setSupports(MEB::OUT_ARG_DfDp,4,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDx,6,MEB::DerivativeSupport(MEB::DERIV_MV_GRADIENT_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDp,4,4,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDp,6,2,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));
  outArgs.setSupports(MEB::OUT_ARG_DgDp,6,4,MEB::DerivativeSupport(MEB::DERIV_MV_JACOBIAN_FORM));

  prototypeOutArgs_ = outArgs;

  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);

  residTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Residual Evaluation");
  jacTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Jacobian Evaluation");

  // Parameter and response support. There exists one parameter and one response.
  for (auto& p : pNames_)
    p = Teuchos::rcp(new Teuchos::Array<std::string>);
  pNames_[0]->push_back("Dummy p(0)");
  pNames_[1]->push_back("Dummy p(1)");
  pNames_[2]->push_back("k");
  pNames_[3]->push_back("Dummy p(3)");
  pNames_[4]->push_back("T_left");
  pMap_ = Teuchos::rcp(new const tpetra_map(1, 0, comm_, Tpetra::LocallyReplicated));
  pSpace_ = ::Thyra::createVectorSpace<Scalar, LO, GO, Node>(pMap_);
  p2_ = ::Thyra::createMember(pSpace_);
  V_S(p2_.ptr(),1.0);
  nominalValues_.set_p(2,p2_);
  p4_ = ::Thyra::createMember(pSpace_);
  V_S(p4_.ptr(),1.0);
  nominalValues_.set_p(4,p4_);

  for (auto& g : gNames_)
    g.clear();
  gNames_[0].push_back("Dummy g(0)");
  gNames_[1].push_back("Dummy g(1)");
  gNames_[2].push_back("Dummy g(2)");
  gNames_[3].push_back("Dummy g(3)");
  gNames_[4].push_back("Constraint: T_right=2");
  gNames_[5].push_back("Dummy g(5)");
  gNames_[6].push_back("Constraint: 2*T_left=T_right");
  gMap_ = Teuchos::rcp(new const tpetra_map(1, 0, comm_, Tpetra::LocallyReplicated));
  gSpace_ = ::Thyra::createVectorSpace<Scalar, LO, GO, Node>(gMap_);
  dgdpMap_ = Teuchos::rcp(new const tpetra_map(1, 0, comm_, Tpetra::LocallyReplicated));
  dgdpSpace_ = ::Thyra::createVectorSpace<Scalar, LO, GO, Node>(dgdpMap_);

  p_name_to_index_["Dummy p(0)"] = std::make_pair(0,0);
  p_name_to_index_["Dummy p(1)"] = std::make_pair(1,0);
  p_name_to_index_["k"] = std::make_pair(2,0);
  p_name_to_index_["Dummy p(3)"] = std::make_pair(3,0);
  p_name_to_index_["T_left"] = std::make_pair(4,0);

  g_name_to_index_["Dummy g(0)"] = std::make_pair(0,0);
  g_name_to_index_["Dummy g(1)"] = std::make_pair(1,0);
  g_name_to_index_["Dummy g(2)"] = std::make_pair(2,0);
  g_name_to_index_["Dummy g(3)"] = std::make_pair(3,0);
  g_name_to_index_["Constraint: T_right=2"] = std::make_pair(4,0);
  g_name_to_index_["Dummy g(5)"] = std::make_pair(5,0);
  g_name_to_index_["Constraint: 2*T_left=T_right"] = std::make_pair(6,0);
}