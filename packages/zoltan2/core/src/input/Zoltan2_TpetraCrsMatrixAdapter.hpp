// // @HEADER
// //
// // ***********************************************************************
// //
// //   Zoltan2: A package of combinatorial algorithms for scientific computing
// //                  Copyright 2012 Sandia Corporation
// //
// // Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// // the U.S. Government retains certain rights in this software.
// //
// // Redistribution and use in source and binary forms, with or without
// // modification, are permitted provided that the following conditions are
// // met:
// //
// // 1. Redistributions of source code must retain the above copyright
// // notice, this list of conditions and the following disclaimer.
// //
// // 2. Redistributions in binary form must reproduce the above copyright
// // notice, this list of conditions and the following disclaimer in the
// // documentation and/or other materials provided with the distribution.
// //
// // 3. Neither the name of the Corporation nor the names of the
// // contributors may be used to endorse or promote products derived from
// // this software without specific prior written permission.
// //
// // THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// // PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// // CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// // LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// // NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// // SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// //
// // Questions? Contact Karen Devine      (kddevin@sandia.gov)
// //                    Erik Boman        (egboman@sandia.gov)
// //                    Siva Rajamanickam (srajama@sandia.gov)
// //
// // ***********************************************************************
// //
// // @HEADER

// /*! \file Zoltan2_XpetraCrsMatrixAdapter.hpp
//     \brief Defines the XpetraCrsMatrixAdapter class.
// */

// #ifndef _ZOLTAN2_TPETRACRSMATRIXADAPTER_HPP_
// #define _ZOLTAN2_TPETRACRSMATRIXADAPTER_HPP_

// #include <Zoltan2_MatrixAdapter.hpp>
// #include <Zoltan2_StridedData.hpp>
// #include <Zoltan2_InputTraits.hpp>
// #include <Zoltan2_PartitioningHelpers.hpp>

// #include <Tpetra_CrsMatrix.hpp>

// #include <vector>

// namespace Zoltan2 {

// //////////////////////////////////////////////////////////////////////////////
// /*!  \brief Provides access for Zoltan2 to Tpetra::CrsMatrix data.

//     \todo we assume FillComplete has been called.  We should support
//                 objects that are not FillCompleted.
//     \todo add RowMatrix

//     The template parameter is the user's input object:
//      \li Tpetra::CrsMatrix

//     The \c scalar_t type, representing use data such as matrix values, is
//     used by Zoltan2 for weights, coordinates, part sizes and
//     quality metrics.
//     Some User types (like Tpetra::CrsMatrix) have an inherent scalar type,
//     and some (like Tpetra::CrsGraph) do not.  For such objects, the scalar
//     type is set by Zoltan2 to \c float.  If you wish to change it to double,
//     set the second template parameter to \c double.

// */

// template <typename User, typename UserCoord = User>
//   class TpetraCrsMatrixAdapter : public TpetraRowMatrix<User,UserCoord> {

// public:
// #ifndef DOXYGEN_SHOULD_SKIP_THIS
//   using scalar_t = typename InputTraits<User>::scalar_t;
//   using lno_t = typename InputTraits<User>::lno_t;
//   using gno_t = typename InputTraits<User>::gno_t;
//   using part_t = typename InputTraits<User>::part_t;
//   using node_t = typename InputTraits<User>::node_t;
//   using offset_t = typename InputTraits<User>::offset_t;
//   using tmatrix_t = Tpetra::CrsMatrix<scalar_t, lno_t, gno_t, node_t>;
//   using device_t = typename node_t::device_type;
//   using host_t = typename Kokkos::HostSpace::memory_space;
//   using user_t = User;
//   using userCoord_t = UserCoord;

//   using Base = MatrixAdapter<User, UserCoord>;
//   using RowMatrix = TpetraRowMatrixAdapter<User, UserCoord>;
//   #endif

//   /*! \brief Constructor
//    *    \param inmatrix The user's Tpetra::CrsMatrix object
//    *    \param nWeightsPerRow If row weights will be provided in setRowWeights(),
//    *        then set \c nWeightsPerRow to the number of weights per row.
//    */
//   TpetraCrsMatrixAdapter(const RCP<const User> &inmatrix,
//                          int nWeightsPerRow=0);

//   /*! \brief Access to user's matrix
//    */
//   RCP<const User> getUserMatrix() const { return this->matrix_; }

//   template <typename Adapter>
//   void applyPartitioningSolution(const User &in, User *&out,
//         const PartitioningSolution<Adapter> &solution) const;

//   template <typename Adapter>
//   void applyPartitioningSolution(const User &in, RCP<User> &out,
//         const PartitioningSolution<Adapter> &solution) const;
// };

// /////////////////////////////////////////////////////////////////
// // Definitions
// /////////////////////////////////////////////////////////////////

//   template <typename User, typename UserCoord>
//   TpetraCrsMatrixAdapter<User,UserCoord>::TpetraCrsMatrixAdapter(
//     const RCP<const User> &inmatrix, int nWeightsPerRow):
//       matrix_(inmatrix), rowMap_(), colMap_(), columnIds_(),
//       nWeightsPerRow_(nWeightsPerRow), rowWeights_(), numNzWeight_(),
//       mayHaveDiagonalEntries(true) {

//   typedef StridedData<lno_t,scalar_t> input_t;
//   using indsHost_t = typename User::nonconst_local_inds_host_view_type;
//   using valsHost_t = typename User::nonconst_values_host_view_type;

//   rowMap_ = matrix_->getRowMap();
//   colMap_ = matrix_->getColMap();

//   size_t nrows = matrix_->getLocalNumRows();
//   size_t nnz = matrix_->getLocalNumEntries();
//   size_t maxnumentries = matrix_->getLocalMaxNumRowEntries(); // Diff from CrsMatrix
//   indsHost_t localColumnIds;

//   offset_.resize(nrows + 1, 0);
//   columnIds_.resize(nnz);
//   values_.resize(nnz);

//   indsHost_t indices("indices", maxnumentries);
//   valsHost_t nzs("nzs", maxnumentries);

//   kOffset_ = Kokkos::View<offset_t *, device_t>(
//       Kokkos::ViewAllocateWithoutInitializing("offset_"), nrows + 1);
//   auto kOffsetHost = Kokkos::create_mirror_view(kOffset_);

//   kColumnIds_ = Kokkos::View<gno_t *, device_t>(
//       Kokkos::ViewAllocateWithoutInitializing("columIds_"), nnz);
//   auto kColumnIdsHost = Kokkos::create_mirror_view(kColumnIds_);

//   kValues_ = Kokkos::View<scalar_t *, device_t>(
//       Kokkos::ViewAllocateWithoutInitializing("values_"), nnz);
//   auto kValuesHost = Kokkos::create_mirror_view(kValues_);

//   // kRowWeights_ = Kokkos::View<scalar_t **, device_t>(
//   //     Kokkos::ViewAllocateWithoutInitializing("rowWeights_"), nWeightsPerRow_,
//   //     nWeightsPerRow_ * nrows);

//   if (nWeightsPerRow_ > 0) {
//     rowWeights_ =
//       arcp(new strided_t[nWeightsPerRow_], 0, nWeightsPerRow_, true);

//     kRowWeights_.resize(nWeightsPerRow_);
//   }

//   for (offset_t i = 0; i < offset_[nrows]; i++) {
//     columnIds_[i] = matrix_->getColMap()->getGlobalElement(localColumnIds[i]);
//   }

//   lno_t next = 0;
//   kOffsetHost(0) = 0;
//   for (size_t i = 0; i < nrows; i++) {
//     lno_t row = i;
//     matrix_->getLocalRowCopy(row, indices, nzs, nnz); // Diff from CrsMatrix
//     for (size_t j = 0; j < nnz; j++) {
//       auto cNzs = nzs[j];
//       values_[next] = cNzs;
//       kValuesHost(next) = cNzs;
//       // TODO - this will be slow
//       //   Is it possible that global columns ids might be stored in order?
//       auto colMapGId = colMap_->getGlobalElement(indices[j]);
//       kColumnIdsHost(next) = colMapGId;
//       columnIds_[next++] = colMapGId;
//     }
//     auto nextOffset = offset_[i] + nnz;
//     offset_[i + 1] = nextOffset;
//     kOffsetHost(i + 1) = nextOffset;
//   }

//   Kokkos::deep_copy(kValues_, kValuesHost);
//   Kokkos::deep_copy(kOffset_, kOffsetHost);
//   Kokkos::deep_copy(kColumnIds_, kColumnIdsHost);

//   if (nWeightsPerRow_ > 0) {
//     rowWeights_ = arcp(new input_t[nWeightsPerRow_], 0, nWeightsPerRow_, true);
//     numNzWeight_ =
//         Kokkos::View<bool *, host_t>("numNzWeight_", nWeightsPerRow_);
//     for (size_t i = 0; i < numNzWeight_.extent(0); ++i) {
//       numNzWeight_(i) = false;
//     }
//   }
// }

// ////////////////////////////////////////////////////////////////////////////
// template <typename User, typename UserCoord>
//   template <typename Adapter>
//     void TpetraCrsMatrixAdapter<User,UserCoord>::applyPartitioningSolution(
//       const User &in, User *&out,
//       const PartitioningSolution<Adapter> &solution) const
// {
//   // Get an import list (rows to be received)
//   size_t numNewRows;
//   ArrayRCP<gno_t> importList;
//   try{
//     numNewRows = Zoltan2::getImportList<Adapter,
//                                         TpetraCrsMatrixAdapter<User,UserCoord> >
//                                        (solution, this, importList);
//   }
//   Z2_FORWARD_EXCEPTIONS;

//   // Move the rows, creating a new matrix.
//   RCP<User> outPtr = doMigration(in, numNewRows,importList.getRawPtr());
//   out = const_cast<User *>(outPtr.get());
//   outPtr.release();
// }

// ////////////////////////////////////////////////////////////////////////////
// template <typename User, typename UserCoord>
//   template <typename Adapter>
//     void TpetraCrsMatrixAdapter<User,UserCoord>::applyPartitioningSolution(
//       const User &in, RCP<User> &out,
//       const PartitioningSolution<Adapter> &solution) const
// {
//   // Get an import list (rows to be received)
//   size_t numNewRows;
//   ArrayRCP<gno_t> importList;
//   try{
//     numNewRows = Zoltan2::getImportList<Adapter,
//                                         TpetraCrsMatrixAdapter<User,UserCoord> >
//                                        (solution, this, importList);
//   }
//   Z2_FORWARD_EXCEPTIONS;

//   // Move the rows, creating a new matrix.
//   out = doMigration(in, numNewRows, importList.getRawPtr());
// }

// }  //namespace Zoltan2

// #endif