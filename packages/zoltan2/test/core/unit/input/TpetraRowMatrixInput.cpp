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

#include <string>

#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_TestHelpers.hpp>
#include <Zoltan2_TpetraRowMatrixAdapter.hpp>
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
using ztrowmatrix_t = Tpetra::RowMatrix<zscalar_t, zlno_t, zgno_t, znode_t>;
using node_t = typename Zoltan2::InputTraits<ztrowmatrix_t>::node_t;
using device_t = typename node_t::device_type;
using rowAdapter_t = Zoltan2::TpetraRowMatrixAdapter<ztrowmatrix_t>;
using crsAdapter_t = Zoltan2::TpetraCrsMatrixAdapter<ztcrsmatrix_t>;
using execspace_t =
    typename rowAdapter_t::ConstWeightsHostView1D::execution_space;

//////////////////////////////////////////////////////////////////////////

template<typename offset_t>
void printMatrix(RCP<const Comm<int> > &comm, zlno_t nrows,
    const zgno_t *rowIds, const offset_t *offsets, const zgno_t *colIds) {
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

template <typename adapter_t, typename matrix_t>
void TestMatrixIds(adapter_t &ia, matrix_t &matrix) {

  using idsHost_t = typename adapter_t::ConstIdsHostView;
  using offsetsHost_t = typename adapter_t::ConstOffsetsHostView;

  using localInds_t =
      typename adapter_t::user_t::nonconst_local_inds_host_view_type;
  using localVals_t =
      typename adapter_t::user_t::nonconst_values_host_view_type;

  const auto nrows = matrix.getLocalNumRows();
  const auto ncols = matrix.getLocalNumEntries();
  const auto maxNumEntries = matrix.getLocalMaxNumRowEntries();

  typename adapter_t::Base::ConstIdsHostView colIdsHost("colIdsHost", ncols);
  typename adapter_t::Base::ConstOffsetsHostView offsHost_("offsHost_",
                                                           nrows + 1);

  localInds_t localColInds("localColInds", maxNumEntries);
  localVals_t localVals("localVals", maxNumEntries);

  for (size_t r = 0; r < nrows; r++) {
    size_t numEntries = 0;
    matrix.getLocalRowCopy(r, localColInds, localVals, numEntries);

    offsHost_(r + 1) = offsHost_(r) + numEntries;
    for (int e = offsHost_(r), i = 0; e < offsHost_(r + 1); e++) {
      colIdsHost(e) = matrix.getColMap()->getGlobalElement(localColInds(i++));
    }
  }

  idsHost_t rowIdsHost;
  ia.getRowIDsHostView(rowIdsHost);

  const auto matrixIDS = matrix.getRowMap()->getLocalElementList();

  Z2_TEST_COMPARE_ARRAYS(matrixIDS, rowIdsHost);
}

//////////////////////////////////////////////////////////////////////////

template <typename adapter_t, typename matrix_t>
void verifyInputAdapter(adapter_t &ia,
                        matrix_t &matrix) {
  using idsDevice_t = typename adapter_t::ConstIdsDeviceView;
  using idsHost_t = typename adapter_t::ConstIdsHostView;
  using offsetsDevice_t = typename adapter_t::ConstOffsetsDeviceView;
  using offsetsHost_t = typename adapter_t::ConstOffsetsHostView;
  using weightsDevice_t = typename adapter_t::WeightsDeviceView1D;
  using weightsHost_t = typename adapter_t::WeightsHostView1D;
  using constWeightsDevice_t = typename adapter_t::ConstWeightsDeviceView1D;
  using constWeightsHost_t = typename adapter_t::ConstWeightsHostView1D;
  using scalarsHost_t = typename adapter_t::ConstScalarsHostView;
  using scalarsDevice_t = typename adapter_t::ConstScalarsDeviceView;

  const auto nrows = ia.getLocalNumIDs();

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
  Z2_TEST_COMPARE_ARRAYS(rowIdsHost, matrix.getRowMap()->getMyGlobalIndices());


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

  /////////////////////////////////
  //// setRowWeightsDevice
  /////////////////////////////////
  Z2_TEST_THROW(ia.setRowWeightsDevice(
                    typename adapter_t::ConstWeightsDeviceView1D{}, 50),
                std::runtime_error);

  weightsDevice_t wgts0("wgts0", nrows);
  Kokkos::parallel_for("wgts0-parallel-for",
                        nrows,
                        KOKKOS_LAMBDA(const int idx) {
                          wgts0(idx) = idx * 2;
                        }
                        );

  Z2_TEST_NOTHROW(ia.setRowWeightsDevice(wgts0, 0));

  weightsDevice_t wgts1("wgts1", nrows);
  Kokkos::parallel_for(
      nrows, KOKKOS_LAMBDA(const int idx) { wgts1(idx) = idx * 3; });

  Z2_TEST_NOTHROW(ia.setRowWeightsDevice(wgts1, 1));

  /////////////////////////////////
  //// getRowWeightsDevice
  /////////////////////////////////
  {
    constWeightsDevice_t weightsDevice;
    Z2_TEST_NOTHROW(ia.getRowWeightsDeviceView(weightsDevice, 0));

    constWeightsHost_t weightsHost;
    Z2_TEST_NOTHROW(ia.getRowWeightsHostView(weightsHost, 0));

    TestDeviceHostView(weightsDevice, weightsHost);

    // /*find good macro*/(wgts0, weightsHost);
  }
  {
    constWeightsDevice_t weightsDevice;
    Z2_TEST_NOTHROW(ia.getRowWeightsDeviceView(weightsDevice, 1));

    constWeightsHost_t weightsHost;
    Z2_TEST_NOTHROW(ia.getRowWeightsHostView(weightsHost, 1));

    TestDeviceHostView(weightsDevice, weightsHost);

    // /*find good macro*/(wgts1, weightsHost);
  }
  {
    constWeightsDevice_t wgtsDevice;
    Z2_TEST_THROW(ia.getRowWeightsDeviceView(wgtsDevice, 2),
                  std::runtime_error);

    constWeightsHost_t wgtsHost;
    Z2_TEST_THROW(ia.getRowWeightsHostView(wgtsHost, 2), std::runtime_error);
  }

  TestMatrixIds(ia, matrix);

}

//////////////////////////////////////////////////////////////////////////

int main (int narg, char *arg[]) {
  using soln_t = Zoltan2::PartitioningSolution<rowAdapter_t>;
  using part_t = rowAdapter_t::part_t;

  Tpetra::ScopeGuard tscope(&narg, &arg);
  const auto comm = Tpetra::getDefaultComm();

  try {
    Teuchos::ParameterList params;
    params.set("input file", "simple");
    params.set("file type", "Chaco");

    auto uinput = rcp(new UserInputForTests(params, comm));

    // Input CRS matrix and row matrix from it
    const auto crsMatrix = uinput->getUITpetraCrsMatrix();
    const auto rowMatrix = rcp_dynamic_cast<ztrowmatrix_t>(crsMatrix);

    const auto nRows = rowMatrix->getLocalNumRows();

    // To test migration in the input adapter we need a Solution object.
    const auto env = rcp(new Zoltan2::Environment(comm));

    const int nWeights = 2; // changed from 2 to 1

    /////////////////////////////////////////////////////////////
    // User object is Tpetra::RowMatrix
    /////////////////////////////////////////////////////////////

    PrintFromRoot("Input adapter for Tpetra::RowMatrix");

    auto tpetraRowMatrixInput = rcp(new rowAdapter_t(rowMatrix, nWeights));

    verifyInputAdapter(*tpetraRowMatrixInput, *rowMatrix);
    std::cout << "test 3" << std::endl;
    part_t *p = new part_t[nRows];
    std::cout << "test 4" << std::endl;
    memset(p, 0, sizeof(part_t) * nRows);
    std::cout << "test 5" << std::endl;
    ArrayRCP<part_t> solnParts(p, 0, nRows, true);
    std::cout << "test 6" << std::endl;

    soln_t solution(env, comm, nWeights);
    solution.setParts(solnParts);
    std::cout << "test 7" << std::endl;

    ztrowmatrix_t *mMigrate = NULL;
    std::cout << "test 8" << std::endl;
    tpetraRowMatrixInput->applyPartitioningSolution(*rowMatrix, mMigrate,
                                                    solution);
    std::cout << "test 9" << std::endl;
    const auto newM = rcp(mMigrate);
    std::cout << "test 14" << std::endl;
    auto cnewM = rcp_const_cast<const ztrowmatrix_t>(newM);
    auto newInput = rcp(new rowAdapter_t(cnewM, nWeights));
    std::cout << "test 15" << std::endl;
    PrintFromRoot("Input adapter for Tpetra::RowMatrix migrated to proc 0");

    verifyInputAdapter(*newInput, *newM);
    std::cout << "test 16" << std::endl;
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  PrintFromRoot("PASS");
}