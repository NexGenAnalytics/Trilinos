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

#include <Zoltan2_PartitioningHelpers.hpp>
#include <Zoltan2_StridedData.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>
#include <Zoltan2_XpetraTraits.hpp>
#include <string>

namespace Zoltan2 {

/*!  \brief Provides access for Zoltan2 to Tpetra::CrsGraph data.

    \todo test for memory alloc failure when we resize a vector
    \todo we assume FillComplete has been called.  Should we support
                objects that are not FillCompleted.
*/

template <typename User, typename UserCoord = User>
class TpetraCrsGraphAdapter : public TpetraRowGraphAdapter<User, UserCoord> {

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
  using RowGraph = TpetraRowGraphAdapter<User, UserCoord>;
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

  /*! \brief Access to user's graph
   */
  RCP<const User> getUserGraph() const { return this->graph_; }

  template <typename Adapter>
  void applyPartitioningSolution(
      const User &in, User *&out,
      const PartitioningSolution<Adapter> &solution) const;

  template <typename Adapter>
  void applyPartitioningSolution(
      const User &in, RCP<User> &out,
      const PartitioningSolution<Adapter> &solution) const;
};

/////////////////////////////////////////////////////////////////
// Definitions
/////////////////////////////////////////////////////////////////

template <typename User, typename UserCoord>
TpetraCrsGraphAdapter<User, UserCoord>::TpetraCrsGraphAdapter(
    const RCP<const User> &graph, int nVtxWgts, int nEdgeWgts)
    : TpetraRowGraphAdapter<User>(nVtxWgts, nEdgeWgts, graph) {

  using localInds_t = typename User::nonconst_local_inds_host_view_type;

  const auto nvtx = graph->getLocalNumRows();
  const auto nedges = graph->getLocalNumEntries();
  // Diff from CrsMatrix
  const auto maxNumEntries = graph->getLocalMaxNumRowEntries();

  // Unfortunately we have to copy the offsets and edge Ids
  // because edge Ids are not usually stored in vertex id order.

  typename Base::ConstIdsHostView adjIdsHost("adjIdsHost_", nedges);
  typename Base::ConstOffsetsHostView offsHost("offsHost_", nvtx + 1);

  localInds_t nbors("nbors", maxNumEntries);

  for (size_t v = 0; v < nvtx; v++) {
    size_t numColInds = 0;
    graph->getLocalRowCopy(v, nbors, numColInds); // Diff from CrsGraph

    offsHost(v + 1) = offsHost(v) + numColInds;
    for (offset_t e = offsHost(v), i = 0; e < offsHost(v + 1); e++) {
      adjIdsHost(e) = graph->getColMap()->getGlobalElement(nbors(i++));
    }
  }

  // local indices
  const auto ajdIdsHost = graph->getLocalIndicesHost();
  // // adjIdsDevice_ = typename Base::ConstIdsDeviceView("adjIdsDevice_",
  // nvtx); auto adjIdsGlobalHost = typename
  // Base::IdsHostView("adjIdsGlobalHost", nvtx); auto colMap =
  // graph_->getColMap();

  // for(int i = 0; i < ajdIdsHost.extent(0); ++i){
  //   adjIdsGlobalHost(i) = colMap->getGlobalElement(ajdIdsHost(i));
  // }

  // Kokkos::deep_copy(adjIdsDevice_, adjIdsGlobalHost);
  // auto tmpView = Kokkos::create_mirror_view_and_copy(typename
  // Base::device_t(), adjIdsGlobalHost); adjIdsDevice_ = tmpView;
  this->offsDevice_ = graph->getLocalRowPtrsDevice();

  if (this->nWeightsPerVertex_ > 0) {

    // should we create unrealying Views aswell?
    this->vertexWeightsDevice_.resize(this->nWeightsPerVertex_);

    this->vertexDegreeWeightsHost_ = typename Base::VtxDegreeHostView(
        "vertexDegreeWeightsHost_", this->nWeightsPerVertex_);

    for (int i = 0; i < this->nWeightsPerVertex_; ++i) {
      this->vertexDegreeWeightsHost_(i) = false;
    }
  }

  this->edgeWeightsDevice_.resize(this->nWeightsPerEdge_);
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
