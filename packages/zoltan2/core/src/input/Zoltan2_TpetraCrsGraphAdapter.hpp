// @HEADER
//
// ***********************************************************************
//
//   Zoltan2: A package of combinatorial algorithms for scientific computing
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Karen Devine      (kddevin@sandia.gov)
//                    Erik Boman        (egboman@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

/*! \file Zoltan2_TpetraCrsGraphAdapter.hpp
    \brief Defines TpetraCrsGraphAdapter class.
*/

#ifndef _ZOLTAN2_TPETRACRSGRAPHADAPTER_HPP_
#define _ZOLTAN2_TPETRACRSGRAPHADAPTER_HPP_

#include "Kokkos_UnorderedMap.hpp"
#include <Zoltan2_GraphAdapter.hpp>
#include <Zoltan2_PartitioningHelpers.hpp>
#include <Zoltan2_StridedData.hpp>
#include <Zoltan2_XpetraTraits.hpp>
#include <string>

namespace Zoltan2 {

/*!  \brief Provides access for Zoltan2 to Tpetra::CrsGraph data.

    \todo test for memory alloc failure when we resize a vector
    \todo we assume FillComplete has been called.  Should we support
                objects that are not FillCompleted.
*/

template <typename User, typename UserCoord = User>
class TpetraCrsGraphAdapter : public GraphAdapter<User, UserCoord> {

public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  using scalar_t = typename InputTraits<User>::scalar_t;
  using offset_t = typename InputTraits<User>::offset_t;
  using lno_t = typename InputTraits<User>::lno_t;
  using gno_t = typename InputTraits<User>::gno_t;
  using part_t = typename InputTraits<User>::part_t;
  using node_t = typename InputTraits<User>::node_t;
  using user_t = User;
  using userCoord_t = UserCoord;

  using Base = GraphAdapter<User, UserCoord>;
#endif

  /*! \brief Constructor for graph with no weights or coordinates.
   *  \param ingraph the Tpetra::CrsGraph
   *  \param numVtxWeights  the number of weights per vertex (default = 0)
   *  \param numEdgeWeights the number of weights per edge  (default = 0)
   *
   * Most adapters do not have RCPs in their interface.  This
   * one does because the user is obviously a Trilinos user.
   */

  TpetraCrsGraphAdapter(const RCP<const User> &graph, int nVtxWeights = 0,
                        int nEdgeWeights = 0);

  /*! \brief Provide a device view of weights for the primary entity type.
   *    \param val A view to the weights for index \c idx.
   *    \param idx A number from 0 to one less than
   *          weight idx specified in the constructor.
   *
   *  The order of the weights should match the order that
   *  entities appear in the input data structure.
   */
  void setWeightsDevice(typename Base::ConstWeightsDeviceView1D val, int idx);

  /*! \brief Provide a host view of weights for the primary entity type.
   *    \param val A view to the weights for index \c idx.
   *    \param idx A number from 0 to one less than
   *          weight idx specified in the constructor.
   *
   *  The order of the weights should match the order that
   *  entities appear in the input data structure.
   */
  void setWeightsHost(typename Base::ConstWeightsHostView1D val, int idx);

  /*! \brief Provide a device view to vertex weights.
   *    \param val A pointer to the weights for index \c idx.
   *    \param idx A number from 0 to one less than
   *          number of vertex weights specified in the constructor.
   *
   *  The order of the vertex weights should match the order that
   *  vertices appear in the input data structure.
   *     \code
   *       TheGraph->getRowMap()->getLocalElementList()
   *     \endcode
   */
  void setVertexWeightsDevice(typename Base::ConstWeightsDeviceView1D val,
                              int idx);

  /*! \brief Provide a host view to vertex weights.
   *    \param val A pointer to the weights for index \c idx.
   *    \param idx A number from 0 to one less than
   *          number of vertex weights specified in the constructor.
   *
   *  The order of the vertex weights should match the order that
   *  vertices appear in the input data structure.
   *     \code
   *       TheGraph->getRowMap()->getLocalElementList()
   *     \endcode
   */
  void setVertexWeightsHost(typename Base::ConstWeightsHostView1D val, int idx);

  /*! \brief Specify an index for which the weight should be
              the degree of the entity
   *    \param idx Zoltan2 will use the entity's
   *         degree as the entity weight for index \c idx.
   */
  void setWeightIsDegree(int idx);

  /*! \brief Specify an index for which the vertex weight should be
              the degree of the vertex
   *    \param idx Zoltan2 will use the vertex's
   *         degree as the vertex weight for index \c idx.
   */
  void setVertexWeightIsDegree(int idx);

  /*! \brief Provide a device view to edge weights.
   *  \param val A pointer to the weights for index \c idx.
   *  \param idx A number from 0 to one less than the number
   *             of edge weights specified in the constructor.
   */
  void setEdgeWeightsDevice(typename Base::ConstWeightsDeviceView1D val,
                            int idx);

  /*! \brief Provide a host view to edge weights.
   *  \param val A pointer to the weights for index \c idx.
   *  \param idx A number from 0 to one less than the
   *             number of edge weights specified in the constructor.
   */
  void setEdgeWeightsHost(typename Base::ConstWeightsHostView1D val, int idx);

  void
  getEdgesDeviceView(typename Base::ConstOffsetsDeviceView &offsets,
                     typename Base::ConstIdsDeviceView &adjIds) const override;

  void getEdgesHostView(typename Base::ConstOffsetsHostView &offsets,
                        typename Base::ConstIdsHostView &adjIds) const override;

  /*! \brief Access to user's graph
   */
  RCP<const User> getUserGraph() const { return graph_; }

  ////////////////////////////////////////////////////
  // The GraphAdapter interface.
  ////////////////////////////////////////////////////

  // TODO:  Assuming rows == objects;
  // TODO:  Need to add option for columns or nonzeros?
  size_t getLocalNumVertices() const override {
    return graph_->getLocalNumRows();
  }

  void getVertexIDsView(const gno_t *&ids) const override {
    ids = NULL;
    if (getLocalNumVertices())
      ids = graph_->getRowMap()->getLocalElementList().getRawPtr();
  }

  void
  getVertexIDsDeviceView(typename Base::ConstIdsDeviceView &ids) const override;

  void
  getVertexIDsHostView(typename Base::ConstIdsHostView &ids) const override;

  size_t getLocalNumEdges() const override {
    return graph_->getLocalNumEntries();
  }

  int getNumWeightsPerVertex() const override { return nWeightsPerVertex_; }

  void
  getVertexWeightsDeviceView(typename Base::ConstWeightsDeviceView1D &weights,
                             int idx = 0) const override;

  void getVertexWeightsHostView(typename Base::ConstWeightsHostView1D &weights,
                                int idx = 0) const override;

  bool useDegreeAsVertexWeight(int idx) const override {
    return vertexDegreeWeightsHost_[idx];
  }

  int getNumWeightsPerEdge() const override { return nWeightsPerEdge_; }

  void getEdgesView(const offset_t *&offsets,
                    const gno_t *&adjIds) const override {
    Z2_THROW_NOT_IMPLEMENTED
  }
  void
  getEdgeWeightsDeviceView(typename Base::ConstWeightsDeviceView1D &weights,
                           int idx = 0) const override;

  void getEdgeWeightsHostView(typename Base::ConstWeightsHostView1D &weights,
                              int idx = 0) const override;

  template <typename Adapter>
  void applyPartitioningSolution(
      const User &in, User *&out,
      const PartitioningSolution<Adapter> &solution) const;

  template <typename Adapter>
  void applyPartitioningSolution(
      const User &in, RCP<User> &out,
      const PartitioningSolution<Adapter> &solution) const;

private:
  RCP<const User> graph_;
  RCP<const Comm<int>> comm_;

  typename Base::ConstIdsDeviceView adjIdsDevice_;
  typename Base::ConstOffsetsDeviceView offsDevice_;

  int nWeightsPerVertex_;
  std::vector<typename Base::ConstWeightsDeviceView1D> vertexWeightsDevice_;
  typename Base::VtxDegreeHostView vertexDegreeWeightsHost_;

  int nWeightsPerEdge_;
  std::vector<typename Base::ConstWeightsDeviceView1D> edgeWeightsDevice_;

  int coordinateDim_;
  ArrayRCP<StridedData<lno_t, scalar_t>> coords_;
};

/////////////////////////////////////////////////////////////////
// Definitions
/////////////////////////////////////////////////////////////////

template <typename User, typename UserCoord>
TpetraCrsGraphAdapter<User, UserCoord>::TpetraCrsGraphAdapter(
    const RCP<const User> &graph, int nVtxWgts, int nEdgeWgts)
    : graph_(graph), nWeightsPerVertex_(nVtxWgts), nWeightsPerEdge_(nEdgeWgts),
      coordinateDim_(0) {

  using localInds_t = typename User::nonconst_local_inds_host_view_type;

  const auto nvtx = graph_->getLocalNumRows();
  const auto nedges = graph_->getLocalNumEntries();
  // Diff from CrsMatrix
  const auto maxNumEntries = graph_->getLocalMaxNumRowEntries();

  // Unfortunately we have to copy the offsets and edge Ids
  // because edge Ids are not usually stored in vertex id order.

  typename Base::ConstIdsHostView adjIdsHost("adjIdsHost_", nedges);
  typename Base::ConstOffsetsHostView offsHost("offsHost_", nvtx + 1);

  localInds_t nbors("nbors", maxNumEntries);

  for (size_t v = 0; v < nvtx; v++) {
    size_t numColInds = 0;
    graph_->getLocalRowCopy(v, nbors, numColInds); // Diff from CrsGraph

    offsHost(v + 1) = offsHost(v) + numColInds;
    for (offset_t e = offsHost(v), i = 0; e < offsHost(v + 1); e++) {
      adjIdsHost(e) = graph_->getColMap()->getGlobalElement(nbors(i++));
    }
  }

  // local indices
  const auto ajdIdsHost = graph_->getLocalIndicesHost();
  // // adjIdsDevice_ = typename Base::ConstIdsDeviceView("adjIdsDevice_", nvtx);
  // auto adjIdsGlobalHost = typename Base::IdsHostView("adjIdsGlobalHost", nvtx);
  // auto colMap = graph_->getColMap();

  // for(int i = 0; i < ajdIdsHost.extent(0); ++i){
  //   adjIdsGlobalHost(i) = colMap->getGlobalElement(ajdIdsHost(i));
  // }

  // Kokkos::deep_copy(adjIdsDevice_, adjIdsGlobalHost);
  // auto tmpView = Kokkos::create_mirror_view_and_copy(typename Base::device_t(), adjIdsGlobalHost);
  // adjIdsDevice_ = tmpView;
  offsDevice_ = graph_->getLocalRowPtrsDevice();

  if (nWeightsPerVertex_ > 0) {

    // should we create unrealying Views aswell?
    vertexWeightsDevice_.resize(nWeightsPerVertex_);

    vertexDegreeWeightsHost_ = typename Base::VtxDegreeHostView(
        "vertexDegreeWeightsHost_", nWeightsPerVertex_);

    for (int i = 0; i < nWeightsPerVertex_; ++i) {
      vertexDegreeWeightsHost_(i) = false;
    }
  }

  edgeWeightsDevice_.resize(nWeightsPerEdge_);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getVertexIDsDeviceView(
    typename Base::ConstIdsDeviceView &ids) const {

  // TODO: Making a  ConstIdsDeviceView LayoutLeft would proably remove the
  //       need of creating tmpIds
  auto idsDevice = graph_->getRowMap()->getMyGlobalIndices();
  auto tmpIds = typename Base::IdsDeviceView("", idsDevice.extent(0));

  Kokkos::deep_copy(tmpIds, idsDevice);

  ids = tmpIds;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getVertexIDsHostView(
    typename Base::ConstIdsHostView &ids) const {
  // TODO: Making a  ConstIdsDeviceView LayoutLeft would proably remove the
  //       need of creating tmpIds
  auto idsDevice = graph_->getRowMap()->getMyGlobalIndices();
  auto tmpIds = typename Base::IdsHostView("", idsDevice.extent(0));

  Kokkos::deep_copy(tmpIds, idsDevice);

  ids = tmpIds;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setWeightsDevice(
    typename Base::ConstWeightsDeviceView1D val, int idx) {
  if (this->getPrimaryEntityType() == GRAPH_VERTEX)
    setVertexWeightsDevice(val, idx);
  else
    setEdgeWeightsDevice(val, idx);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setWeightsHost(
    typename Base::ConstWeightsHostView1D val, int idx) {
  if (this->getPrimaryEntityType() == GRAPH_VERTEX)
    setVertexWeightsHost(val, idx);
  else
    setEdgeWeightsHost(val, idx);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setVertexWeightsDevice(
    typename Base::ConstWeightsDeviceView1D weights, int idx) {

  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid vertex weight index: " + std::to_string(idx));

  vertexWeightsDevice_[idx] = weights;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setVertexWeightsHost(
    typename Base::ConstWeightsHostView1D weightsHost, int idx) {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid vertex weight index: " + std::to_string(idx));

  auto weightsDevice = Kokkos::create_mirror_view_and_copy(
      typename Base::device_t(), weightsHost);
  vertexWeightsDevice_[idx] = weightsDevice;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getVertexWeightsDeviceView(
    typename Base::ConstWeightsDeviceView1D &weights, int idx) const {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid vertex weight index.");
  weights = vertexWeightsDevice_.at(idx);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getVertexWeightsHostView(
    typename Base::ConstWeightsHostView1D &weights, int idx) const {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid vertex weight index.");
  const auto weightsDevice = vertexWeightsDevice_.at(idx);
  weights = Kokkos::create_mirror_view(weightsDevice);
  Kokkos::deep_copy(weights, weightsDevice);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setWeightIsDegree(int idx) {
  AssertCondition(this->getPrimaryEntityType() == GRAPH_VERTEX,
                  "setWeightIsNumberOfNonZeros is supported only for vertices");

  setVertexWeightIsDegree(idx);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setVertexWeightIsDegree(int idx) {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid vertex weight index: " + std::to_string(idx));

  vertexDegreeWeightsHost_[idx] = true;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setEdgeWeightsDevice(
    typename Base::ConstWeightsDeviceView1D weights, int idx) {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid edge weight index.");

  edgeWeightsDevice_[idx] = weights;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::setEdgeWeightsHost(
    typename Base::ConstWeightsHostView1D weightsHost, int idx) {
  AssertCondition((idx >= 0) and (idx < nWeightsPerVertex_),
                  "Invalid edge weight index.");

  auto weightsDevice = Kokkos::create_mirror_view_and_copy(
      typename Base::device_t(), weightsHost);
  edgeWeightsDevice_[idx] = weightsDevice;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getEdgeWeightsDeviceView(
    typename Base::ConstWeightsDeviceView1D &weights, int idx) const {
  weights = edgeWeightsDevice_.at(idx);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getEdgeWeightsHostView(
    typename Base::ConstWeightsHostView1D &weights, int idx) const {
  const auto weightsDevice = edgeWeightsDevice_.at(idx);
  weights = Kokkos::create_mirror_view(weightsDevice);
  Kokkos::deep_copy(weights, weightsDevice);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getEdgesDeviceView(
    typename Base::ConstOffsetsDeviceView &offsets,
    typename Base::ConstIdsDeviceView &adjIds) const {

  offsets = offsDevice_;
  adjIds = adjIdsDevice_;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
void TpetraCrsGraphAdapter<User, UserCoord>::getEdgesHostView(
    typename Base::ConstOffsetsHostView &offsets,
    typename Base::ConstIdsHostView &adjIds) const {
  adjIds = Kokkos::create_mirror_view(adjIdsDevice_);
  Kokkos::deep_copy(adjIds, adjIdsDevice_);

  offsets = Kokkos::create_mirror_view(offsDevice_);
  Kokkos::deep_copy(offsets, offsDevice_);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
template <typename Adapter>
void TpetraCrsGraphAdapter<User, UserCoord>::applyPartitioningSolution(
    const User &in, User *&out,
    const PartitioningSolution<Adapter> &solution) const {
  // Get an import list (rows to be received)
  size_t numNewVtx;
  ArrayRCP<gno_t> importList;
  try {
    numNewVtx =
        Zoltan2::getImportList<Adapter, TpetraCrsGraphAdapter<User, UserCoord>>(
            solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new graph.
  RCP<User> outPtr =
      XpetraTraits<User>::doMigration(in, numNewVtx, importList.getRawPtr());
  out = outPtr.get();
  outPtr.release();
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
template <typename Adapter>
void TpetraCrsGraphAdapter<User, UserCoord>::applyPartitioningSolution(
    const User &in, RCP<User> &out,
    const PartitioningSolution<Adapter> &solution) const {
  // Get an import list (rows to be received)
  size_t numNewVtx;
  ArrayRCP<gno_t> importList;
  try {
    numNewVtx =
        Zoltan2::getImportList<Adapter, TpetraCrsGraphAdapter<User, UserCoord>>(
            solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new graph.
  out = XpetraTraits<User>::doMigration(in, numNewVtx, importList.getRawPtr());
}

} // namespace Zoltan2

#endif
