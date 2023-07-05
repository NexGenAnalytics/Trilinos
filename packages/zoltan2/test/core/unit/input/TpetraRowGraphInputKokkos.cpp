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
//
// Basic testing of Zoltan2::TpetraRowGraphAdapter
/*!  \file TpetraRowGraphAdapter.cpp
 *   \brief Test of Zoltan2::TpetraRowGraphAdapter class.
 *  \todo add weights and coordinates
 */

#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_TestHelpers.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_UnitTestHarness.hpp>

using Teuchos::Comm;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;

using ztcrsgraph_t = Tpetra::CrsGraph<zlno_t, zgno_t, znode_t>;
using ztrowgraph_t = Tpetra::RowGraph<zlno_t, zgno_t, znode_t>;
using node_t = typename Zoltan2::InputTraits<ztrowgraph_t>::node_t;
using device_t = typename node_t::device_type;
using adapter_t = Zoltan2::TpetraRowGraphAdapter<ztrowgraph_t>;

template <typename offset_t>
void printGraph(RCP<const Comm<int>> &comm, zlno_t nvtx, const zgno_t *vtxIds,
                const offset_t *offsets, const zgno_t *edgeIds) {
  int rank = comm->getRank();
  int nprocs = comm->getSize();
  comm->barrier();
  for (int p = 0; p < nprocs; p++) {
    if (p == rank) {
      std::cout << rank << ":" << std::endl;
      for (zlno_t i = 0; i < nvtx; i++) {
        std::cout << " vertex " << vtxIds[i] << ": ";
        for (offset_t j = offsets[i]; j < offsets[i + 1]; j++) {
          std::cout << edgeIds[j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout.flush();
    }
    comm->barrier();
  }
  comm->barrier();
}

template <typename User>
void verifyInputAdapter(Zoltan2::TpetraRowGraphAdapter<User> &ia,
                        ztrowgraph_t &graph) {
  const auto comm = graph.getComm();
  int fail = 0, gfail = 0;
  const auto nVtx = ia.getLocalNumIDs();

  auto &out = std::cout;
  bool success = true;

  TEST_EQUALITY(ia.getLocalNumVertices(), graph.getLocalNumRows());
  TEST_EQUALITY(ia.getLocalNumEdges(), graph.getLocalNumEntries());

  /////////////////////////////////
  //// getVertexIdsView
  /////////////////////////////////

  typename adapter_t::ConstIdsDeviceView vtxIdsDevice;
  ia.getVertexIDsDeviceView(vtxIdsDevice);
  typename adapter_t::ConstIdsHostView vtxIdsHost;
  ia.getVertexIDsHostView(vtxIdsHost);

  TestDeviceHostView(vtxIdsDevice, vtxIdsHost);

  /////////////////////////////////
  //// getEdgesView
  /////////////////////////////////

  typename adapter_t::ConstIdsDeviceView adjIdsDevice;
  typename adapter_t::ConstOffsetsDeviceView offsetsDevice;

  ia.getEdgesDeviceView(offsetsDevice, adjIdsDevice);

  typename adapter_t::ConstIdsHostView adjIdsHost;
  typename adapter_t::ConstOffsetsHostView offsetsHost;
  ia.getEdgesHostView(offsetsHost, adjIdsHost);

  TestDeviceHostView(adjIdsDevice, adjIdsHost);
  TestDeviceHostView(offsetsDevice, offsetsHost);

  /////////////////////////////////
  //// setVertexWeightsDevice
  /////////////////////////////////
  TEST_THROW(ia.setVertexWeightsDevice(
                 typename adapter_t::ConstWeightsDeviceView1D{}, 50),
             std::runtime_error);

  typename adapter_t::WeightsDeviceView1D wgts("wgts", nVtx);
  Kokkos::parallel_for(
      nVtx, KOKKOS_LAMBDA(const int idx) { wgts(idx) = idx * 2; });

  TEST_NOTHROW(ia.setVertexWeightsDevice(wgts, 0));

  Kokkos::parallel_for(
      nVtx, KOKKOS_LAMBDA(const int idx) { wgts(idx) = idx * 3; });

  TEST_NOTHROW(ia.setVertexWeightsDevice(wgts, 1));
  {
    typename adapter_t::ConstWeightsDeviceView1D weightsDevice;
    TEST_NOTHROW(ia.getVertexWeightsDeviceView(weightsDevice, 0));

    typename adapter_t::ConstWeightsHostView1D weightsHost;
    TEST_NOTHROW(ia.getVertexWeightsHostView(weightsHost, 0));

    TestDeviceHostView(weightsDevice, weightsHost);
  }
  {
    typename adapter_t::ConstWeightsDeviceView1D weightsDevice;
    TEST_NOTHROW(ia.getVertexWeightsDeviceView(weightsDevice, 1));

    typename adapter_t::ConstWeightsHostView1D weightsHost;
    TEST_NOTHROW(ia.getVertexWeightsHostView(weightsHost, 1));

    TestDeviceHostView(weightsDevice, weightsHost);
  }
}

int main(int narg, char *arg[]) {
  using soln_t = Zoltan2::PartitioningSolution<adapter_t>;
  using part_t = adapter_t::part_t;

  Tpetra::ScopeGuard tscope(&narg, &arg);
  const auto comm = Tpetra::getDefaultComm();

  auto rank = comm->getRank();

  try {
    Teuchos::ParameterList params;
    params.set("input file", "simple");
    params.set("file type", "Chaco");

    auto uinput = rcp(new UserInputForTests(params, comm));

    // Input crs graph and row graph cast from it.
    const auto crsGraph = uinput->getUITpetraCrsGraph();
    const auto rowGraph = rcp_dynamic_cast<ztrowgraph_t>(crsGraph);

    const auto nvtx = rowGraph->getLocalNumRows();

    // To test migration in the input adapter we need a Solution object.
    const auto env = rcp(new Zoltan2::Environment(comm));

    const int nWeights = 2;

    part_t *p = new part_t[nvtx];
    memset(p, 0, sizeof(part_t) * nvtx);
    ArrayRCP<part_t> solnParts(p, 0, nvtx, true);

    soln_t solution(env, comm, nWeights);
    solution.setParts(solnParts);

    /////////////////////////////////////////////////////////////
    // User object is Tpetra::RowGraph
    /////////////////////////////////////////////////////////////

    PrintFromRoot("Input adapter for Tpetra::RowGraph");

    auto tpetraRowGraphInput = rcp(new adapter_t(rowGraph, nWeights));

    verifyInputAdapter<ztrowgraph_t>(*tpetraRowGraphInput, *crsGraph);

    ztrowgraph_t *mMigrate = NULL;
    tpetraRowGraphInput->applyPartitioningSolution(*crsGraph, mMigrate,
                                                   solution);
    const auto newG = rcp(mMigrate);

    auto cnewG = rcp_const_cast<const ztrowgraph_t>(newG);
    auto newInput = rcp(new adapter_t(cnewG, nWeights));

    PrintFromRoot("Input adapter for Tpetra::RowGraph migrated to proc 0");

    verifyInputAdapter<ztrowgraph_t>(*newInput, *newG);

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    PrintFromRoot("FAIL");
  }

  PrintFromRoot("PASS");
}
