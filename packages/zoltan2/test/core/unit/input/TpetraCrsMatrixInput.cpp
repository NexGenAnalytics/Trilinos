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
// Basic testing of Zoltan2::TpetraCrsMatrixAdapter
/*!  \file TpetraCrsMatrixAdapter.cpp
 *   \brief Test of Zoltan2::TpetraCrsMatrixAdapter class.
 *   \todo add weights and coordinates
 */

#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_TestHelpers.hpp>
#include <Zoltan2_TpetraCrsMatrixAdapter.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <cstdlib>
#include <stdexcept>

using Teuchos::Comm;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;

using ztcrsmatrix_t = Tpetra::CrsMatrix<zscalar_t, zlno_t, zgno_t, znode_t>;
using node_t = typename Zoltan2::InputTraits<ztcrsmatrix_t>::node_t;
using device_t = typename node_t::device_type;
using adapter_t = Zoltan2::TpetraCrsMatrixAdapter<ztcrsmatrix_t>;

//////////////////////////////////////////////////////////////////////////

// is this necessary?
template<typename offset_t>
void printMatrix(RCP<const Comm<int> > &comm, zlno_t nrows,
    const zgno_t *rowIds, const offset_t *offsets, const zgno_t *colIds)
{
  int rank = comm->getRank();
  int nprocs = comm->getSize();
  comm->barrier();
  for (int p=0; p < nprocs; p++){
    if (p == rank){
      std::cout << rank << ":" << std::endl;
      for (zlno_t i=0; i < nrows; i++){
        std::cout << " row " << rowIds[i] << ": ";
        for (offset_t j=offsets[i]; j < offsets[i+1]; j++){
          std::cout << colIds[j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout.flush();
    }
    comm->barrier();
  }
  comm->barrier();
}

//////////////////////////////////////////////////////////////////////////

template <typename User>
void verifyInputAdapter(Zoltan2::TpetraCrsMatrixAdapter<User> &ia,
                        ztcrsmatrix_t &matrix) {
  using idsDevice_t = typename adapter_t::ConstIdsDeviceView;
  using idsHost_t = typename adapter_t::ConstIdsHostView;
  using offsetsDevice_t = typename adapter_t::ConstOffsetsDeviceView;
  using offsetsHost_t = typename adapter_t::ConstOffsetsHostView;
  using weightsDevice_t = typename adapter_t::WeightsDeviceView;
  using weightsHost_t = typename adapter_t::WeightsHostView;
  using scalarsHost_t = typename adapter_t::ConstScalarsHostView;
  using scalarsDevice_t = typename adapter_t::ConstScalarsDeviceView;

  const auto nrows = ia.getLocalNumIDs(); // check if needed

  Z2_TEST_EQUALITY(ia.getLocalNumRows(), matrix.getLocalNumRows());
  Z2_TEST_EQUALITY(ia.getLocalNumColumns(), matrix.getLocalNumCols());
  Z2_TEST_EQUALITY(ia.getLocalNumEntries(), matrix.getLocalNumEntries());

  /////////////////////////////////
  //// getRowIDsView
  /////////////////////////////////

  idsDevice_t rowIdsDevice;
  ia.getRowIDsDeviceView(rowIdsDevice);
  idsHost_t rowIdsHost;
  ia.getRowIDsHostView(rowIdsHost);

  TestDeviceHostView(rowIdsDevice, rowIdsHost);

  // still need to test if equal to expected

  /////////////////////////////////
  //// getCRSView
  /////////////////////////////////

  offsetsDevice_t offsetsDevice;
  idsDevice_t colIdsDevice;
  scalarsDevice_t valuesDevice;
  ia.getCRSDeviceView(offsetsDevice, colIdsDevice, valuesDevice);

  offsetsHost_t offsetsHost;
  idsHost_t colIdsHost;
  scalarsHost_t valuesHost;
  ia.getCRSHostView(offsetsHost, colIdsHost, valuesHost);

  TestDeviceHostView(offsetsDevice, offsetsHost);
  TestDeviceHostView(colIdsDevice, colIdsHost);
  TestDeviceHostView(valuesDevice, valuesHost);

  // TestDeviceHostView(valuesDevice, offsetsHost); // cite fail location?

  // still need to test if equal to expected

  /////////////////////////////////
  //// getRowWeightsView
  /////////////////////////////////

  weightsDevice_t weightsDevice;
  Z2_TEST_NOTHROW(ia.getRowWeightsDeviceView(weightsDevice));

  weightsHost_t weightsHost;
  Z2_TEST_NOTHROW(ia.getRowWeightsHostView(weightsHost));

  // compare Views element-wise
  std::cout << "Comparing hostView == mirrorDevice ... ";
  auto h_weightsDevice = Kokkos::create_mirror_view(weightsDevice);
  Kokkos::deep_copy(h_weightsDevice, weightsDevice);

  for (size_t i=0; i < h_weightsDevice.extent(0); i++) {
    for (size_t j=0; j < h_weightsDevice.extent(1); i++) {
      assert(h_weightsDevice[i][j] == weightsHost[i][j]);
    }
  }
  std::cout << "passed" << std::endl;

  // still need to test if equal to expected
}

//////////////////////////////////////////////////////////////////////////

int main (int narg, char *arg[]) {
  using soln_t = Zoltan2::PartitioningSolution<adapter_t>;
  using part_t = adapter_t::part_t;

  Tpetra::ScopeGuard tscope(&narg, &arg);
  const auto comm = Tpetra::getDefaultComm();

  try {
    Teuchos::ParameterList params;
    params.set("input file", "simple");
    params.set("file type", "Chaco");

    auto uinput = rcp(new UserInputForTests(params, comm));

    // Input CRS matrix
    const auto crsMatrix = uinput->getUITpetraCrsMatrix();
    const auto nrows = crsMatrix->getLocalNumRows();

    // To test migration in the input adapter we need a Solution object.
    const auto env = rcp(new Zoltan2::Environment(comm));

    const int nWeights = 1; // changed from 2 to 1
    part_t *p = new part_t[nrows];
    memset(p, 0, sizeof(part_t) * nrows);
    ArrayRCP<part_t> solnParts(p, 0, nrows, true);

    soln_t solution(env, comm, nWeights);
    solution.setParts(solnParts);

    /////////////////////////////////////////////////////////////
    // User object is Tpetra::CrsMatrix
    /////////////////////////////////////////////////////////////

    PrintFromRoot("Input adapter for Tpetra::CrsMatrix");

    auto ctM = rcp_const_cast<ztcrsmatrix_t>(crsMatrix);
    auto tpetraCrsMatrixInput = rcp(new adapter_t(ctM, nWeights));

    verifyInputAdapter<ztcrsmatrix_t>(*tpetraCrsMatrixInput, *crsMatrix);

    ztcrsmatrix_t *mMigrate = NULL;
    tpetraCrsMatrixInput->applyPartitioningSolution(*crsMatrix, mMigrate,
                                                    solution);
    const auto newM = rcp(mMigrate);

    // auto cnewM = rcp_const_cast<const ztcrsmatrix_t>(newM);
    auto newInput = rcp(new adapter_t(newM, nWeights));

    PrintFromRoot("Input adapter for Tpetra::CrsMatrix migrated to proc 0");

    verifyInputAdapter<ztcrsmatrix_t>(*newInput, *newM);

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  PrintFromRoot("PASS");
}