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

/*! \file Zoltan2_XpetraCrsMatrixAdapter.hpp
    \brief Defines the XpetraCrsMatrixAdapter class.
*/

#ifndef _ZOLTAN2_TPETRACRSMATRIXADAPTER_HPP_
#define _ZOLTAN2_TPETRACRSMATRIXADAPTER_HPP_

#include <Zoltan2_MatrixAdapter.hpp>
#include <Zoltan2_StridedData.hpp>
#include <Zoltan2_XpetraTraits.hpp> // maybe delete
#include <Zoltan2_InputTraits.hpp> // maybe
#include <Zoltan2_PartitioningHelpers.hpp>
#include <Zoltan2_TpetraRowMatrixAdapter.hpp> // maybe for doMigration

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_RowMatrix.hpp> // maybe

namespace Zoltan2 {

//////////////////////////////////////////////////////////////////////////////
/*!  \brief Provides access for Zoltan2 to Tpetra::CrsMatrix data.

    \todo we assume FillComplete has been called.  We should support
                objects that are not FillCompleted.
    \todo add RowMatrix

    The template parameter is the user's input object:
     \li Tpetra::CrsMatrix

    The \c scalar_t type, representing use data such as matrix values, is
    used by Zoltan2 for weights, coordinates, part sizes and
    quality metrics.
    Some User types (like Tpetra::CrsMatrix) have an inherent scalar type,
    and some
    (like Tpetra::CrsGraph) do not.  For such objects, the scalar type is
    set by Zoltan2 to \c float.  If you wish to change it to double, set
    the second template parameter to \c double.

*/

template <typename User, typename UserCoord=User>
  class TpetraCrsMatrixAdapter : public MatrixAdapter<User,UserCoord> {
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  using scalar_t = typename InputTraits<User>::scalar_t;
  using lno_t = typename InputTraits<User>::lno_t;
  using gno_t = typename InputTraits<User>::gno_t;
  using part_t = typename InputTraits<User>::part_t;
  using node_t = typename InputTraits<User>::node_t;
  using offset_t = typename InputTraits<User>::offset_t;
  using tmatrix_t = Tpetra::CrsMatrix<scalar_t, lno_t, gno_t, node_t>;
  using device_t = typename node_t::device_type; // for Kokkos-ification
  using host_t = typename Kokkos::HostSpace::memory_space; // Kokkos-ification
  using user_t = User;
  using userCoord_t = UserCoord;
#endif

  /*! \brief Constructor
   *    \param inmatrix The user's Epetra, Tpetra, or Xpetra CrsMatrix object
   *    \param nWeightsPerRow If row weights will be provided in setRowWeights(),
   *        the set \c nWeightsPerRow to the number of weights per row.
   */
  TpetraCrsMatrixAdapter(const RCP<const User> &inmatrix,
                         int nWeightsPerRow=0);

  /*! \brief Specify a weight for each entity of the primaryEntityType.
   *    \param weightVal A pointer to the weights for this index.
   *    \stride          A stride to be used in reading the values.  The
   *        index \c idx weight for entity \k should be found at
   *        <tt>weightVal[k*stride]</tt>.
   *    \param idx  A value between zero and one less that the \c nWeightsPerRow
   *                  argument to the constructor.
   *
   * The order of weights should correspond to the order of the primary
   * entity type; see, e.g.,  setRowWeights below.
   */

  void setWeights(const scalar_t *weightVal, int stride, int idx = 0);
  void setWeights(Kokkos::View<scalar_t **, device_t> &weights);

  /*! \brief Specify a weight for each row.
   *    \param weightVal A pointer to the weights for this index.
   *    \stride          A stride to be used in reading the values.  The
   *        index \c idx weight for row \k should be found at
   *        <tt>weightVal[k*stride]</tt>.
   *    \param idx  A value between zero and one less that the \c nWeightsPerRow
   *                  argument to the constructor.
   *
   * The order of weights should correspond to the order of rows
   * returned by
   *   \code
   *       theMatrix->getRowMap()->getLocalElementList();
   *   \endcode
   */

  void setRowWeights(const scalar_t *weightVal, int stride, int idx = 0);
  void setRowWeights(Kokkos::View<scalar_t **, device_t> &weights);

  /*! \brief Specify an index for which the weight should be
              the degree of the entity
   *    \param idx Zoltan2 will use the entity's
   *         degree as the entity weight for index \c idx.
   */
  void setWeightIsDegree(int idx);

  /*! \brief Specify an index for which the row weight should be
              the global number of nonzeros in the row
   *    \param idx Zoltan2 will use the global number of nonzeros in a row
   *         as the row weight for index \c idx.
   */
  void setRowWeightIsNumberOfNonZeros(int idx);

  ////////////////////////////////////////////////////
  // The MatrixAdapter interface.
  ////////////////////////////////////////////////////

  size_t getLocalNumRows() const {
    return matrix_->getLocalNumRows();
  }

  size_t getLocalNumColumns() const {
    return matrix_->getLocalNumCols();
  }

  size_t getLocalNumEntries() const {
    return matrix_->getLocalNumEntries();
  }

  bool CRSViewAvailable() const { return true; }

  void getRowIDsView(const gno_t *&rowIds) const override {
    ArrayView<const gno_t> rowView = rowMap_->getLocalElementList();
    rowIds = rowView.getRawPtr();
  }

  void getRowIDsHostView(
      typename BaseAdapter<User>::ConstIdsHostView &rowIds) const override {
    auto kRowIds = rowMap_->getMyGlobalIndices();
    auto rowIdsHost = Kokkos::create_mirror_view(kRowIds);
    Kokkos::deep_copy(rowIdsHost, kRowIds);
    rowIds = rowIdsHost;
  }

  // should we define kRowIds with other member variables?
  void getRowIDsDeviceView(
      typename BaseAdapter<User>::ConstIdsDeviceView &rowIds) const override {
    auto kRowIds = rowMap_->getMyGlobalIndices();
    auto rowIdsDevice = Kokkos::create_mirror_view_and_copy(
      device_t(), kRowIds);
    rowIds = rowIdsDevice;
  }

  void getCRSView(ArrayRCP<const offset_t> &offsets,
                  ArrayRCP<const gno_t> &colIds) const {
    offsets = offset_;
    colIds = columnIds_;
  }

  void getCRSHostView(
      typename BaseAdapter<User>::ConstOffsetsHostView &offsets,
      typename BaseAdapter<User>::ConstIdsHostView &colIds) const override {
    auto hostOffsets = Kokkos::create_mirror_view(kOffset_);
    Kokkos::deep_copy(hostOffsets, kOffset_);
    offsets= hostOffsets;

    auto hostColIds = Kokkos::create_mirror_view(kColumnIds_);
    Kokkos::deep_copy(hostColIds, kColumnIds_);
    colIds = hostColIds;
  }

  void getCRSDeviceView(
      typename BaseAdapter<User>::ConstOffsetsDeviceView &offsets,
      typename BaseAdapter<User>::ConstIdsDeviceView &colIds) const override {
    offsets = kOffset_;
    colIds = kColumnIds_;
  }

  void getCRSView(ArrayRCP<const offset_t> &offsets,
                  ArrayRCP<const gno_t> &colIds,
                  ArrayRCP<const scalar_t> &values) const {
    offsets = offset_;
    colIds = columnIds_;
    values = values_;
  }

  void getCRSHostView(
      typename BaseAdapter<User>::ConstOffsetsHostView &offsets,
      typename BaseAdapter<User>::ConstIdsHostView &colIds,
      typename BaseAdapter<User>::ConstScalarsHostView &values) const override {
    auto hostOffsets = Kokkos::create_mirror_view(kOffset_);
    Kokkos::deep_copy(hostOffsets, kOffset_);
    offsets = hostOffsets;

    auto hostColIds = Kokkos::create_mirror_view(kColumnIds_);
    Kokkos::deep_copy(hostColIds, kColumnIds_);
    colIds = hostColIds;

    auto hostValues = Kokkos::create_mirror_view(kValues_);
    Kokkos::deep_copy(hostValues, kValues_);
    values = hostValues;
  }

  void
  getCRSDeviceView(typename BaseAdapter<User>::ConstOffsetsDeviceView &offsets,
                   typename BaseAdapter<User>::ConstIdsDeviceView &colIds,
                   typename BaseAdapter<User>::ConstScalarsDeviceView &values)
      const override {
    offsets = kOffset_;
    colIds = kColumnIds_;
    values = kValues_;
  }

  int getNumWeightsPerRow() const { return nWeightsPerRow_; }

  void getRowWeightsView(const scalar_t *&weights, int &stride,
                           int idx = 0) const {
    if(idx<0 || idx >= nWeightsPerRow_) {
      std::ostringstream emsg;
      emsg << __FILE__ << ":" << __LINE__ << "  Invalid row weight index "
          << idx << std::endl;
      throw std::runtime_error(emsg.str());
    }

    size_t length;
    rowWeights_[idx].getStridedList(length, weights, stride);
  }

  void getRowWeightsHostView(
      typename BaseAdapter<User>::WeightsHostView &weights) const {
    auto hostWeight = Kokkos::create_mirror_view(kRowWeights_);
    Kokkos::deep_copy(hostWeight, kRowWeights_);
    weights = hostWeight;
  }

  void getRowWeightsDeviceView(
      typename BaseAdapter<User>::WeightsDeviceView &weights) const {
    weights = kRowWeights_;
  }

  bool useNumNonzerosAsRowWeight(int idx) const { return numNzWeight_[idx];}

  template <typename Adapter>
    void applyPartitioningSolution(const User &in, User *&out,
         const PartitioningSolution<Adapter> &solution) const;

  template <typename Adapter>
    void applyPartitioningSolution(const User &in, RCP<User> &out,
         const PartitioningSolution<Adapter> &solution) const;

private:

  RCP<const tmatrix_t> matrix_;
  RCP<const Tpetra::Map<lno_t, gno_t, node_t> > rowMap_;
  RCP<const Tpetra::Map<lno_t, gno_t, node_t> > colMap_;
  lno_t base_;

  ArrayRCP<offset_t> offset_;
  ArrayRCP<gno_t> columnIds_;
  ArrayRCP<scalar_t> values_;
  ArrayRCP<StridedData<lno_t, scalar_t>> rowWeights_;
  int nWeightsPerRow_;

  // New Kokkos Type
  Kokkos::View<offset_t *, device_t> kOffset_;
  Kokkos::View<gno_t *, device_t> kColumnIds_;
  Kokkos::View<scalar_t *, device_t> kValues_;
  Kokkos::View<scalar_t **, device_t> kRowWeights_;
  Kokkos::View<bool *, host_t> numNzWeight_;

  bool mayHaveDiagonalEntries;
  // is there a better way to use doMigration? Put in InputTraits?
  // (right now, it's declared in both Crs and Row)
  RCP<User> doMigration(const User &from, size_t numLocalRows,
                        const gno_t *myNewRows) const;
};

/////////////////////////////////////////////////////////////////
// Definitions
/////////////////////////////////////////////////////////////////

template <typename User, typename UserCoord>
  TpetraCrsMatrixAdapter<User,UserCoord>::TpetraCrsMatrixAdapter(
    const RCP<const User> &inmatrix, int nWeightsPerRow):
      matrix_(inmatrix), rowMap_(), colMap_(), columnIds_(),
      nWeightsPerRow_(nWeightsPerRow), rowWeights_(), numNzWeight_(),
      mayHaveDiagonalEntries(true) {

  typedef StridedData<lno_t,scalar_t> input_t;
  using indsHost_t = typename User::nonconst_local_inds_host_view_type;
  using valsHost_t = typename User::nonconst_values_host_view_type;

  rowMap_ = matrix_->getRowMap();
  colMap_ = matrix_->getColMap();

  size_t nrows = matrix_->getLocalNumRows();
  size_t nnz = matrix_->getLocalNumEntries();
  size_t maxnumentries = matrix_->getLocalMaxNumRowEntries(); // Diff from CrsMatrix
  indsHost_t localColumnIds;

  offset_.resize(nrows + 1, 0);
  columnIds_.resize(nnz);
  values_.resize(nnz);

  indsHost_t indices("indices", maxnumentries);
  valsHost_t nzs("nzs", maxnumentries);

  kOffset_ = Kokkos::View<offset_t *, device_t>(
      Kokkos::ViewAllocateWithoutInitializing("offset_"), nrows + 1);
  auto kOffsetHost = Kokkos::create_mirror_view(kOffset_);

  kColumnIds_ = Kokkos::View<gno_t *, device_t>(
      Kokkos::ViewAllocateWithoutInitializing("columIds_"), nnz);
  auto kColumnIdsHost = Kokkos::create_mirror_view(kColumnIds_);

  kValues_ = Kokkos::View<scalar_t *, device_t>(
      Kokkos::ViewAllocateWithoutInitializing("values_"), nnz);
  auto kValuesHost = Kokkos::create_mirror_view(kValues_);

  kRowWeights_ = Kokkos::View<scalar_t **, device_t>(
      Kokkos::ViewAllocateWithoutInitializing("rowWeights_"), nWeightsPerRow_,
      nWeightsPerRow_ * nrows);

  for (offset_t i = 0; i < offset_[nrows]; i++) {
    columnIds_[i] = matrix_->getColMap()->getGlobalElement(localColumnIds[i]);
  }

  lno_t next = 0;
  kOffsetHost(0) = 0;
  for (size_t i = 0; i < nrows; i++) {
    lno_t row = i;
    matrix_->getLocalRowCopy(row, indices, nzs, nnz); // Diff from CrsMatrix
    for (size_t j = 0; j < nnz; j++) {
      auto cNzs = nzs[j];
      values_[next] = cNzs;
      kValuesHost(next) = cNzs;
      // TODO - this will be slow
      //   Is it possible that global columns ids might be stored in order?
      auto colMapGId = colMap_->getGlobalElement(indices[j]);
      kColumnIdsHost(next) = colMapGId;
      columnIds_[next++] = colMapGId;
    }
    auto nextOffset = offset_[i] + nnz;
    offset_[i + 1] = nextOffset;
    kOffsetHost(i + 1) = nextOffset;
  }

  Kokkos::deep_copy(kValues_, kValuesHost);
  Kokkos::deep_copy(kOffset_, kOffsetHost);
  Kokkos::deep_copy(kColumnIds_, kColumnIdsHost);

  if (nWeightsPerRow_ > 0) {
    rowWeights_ = arcp(new input_t[nWeightsPerRow_], 0, nWeightsPerRow_, true);
    numNzWeight_ =
        Kokkos::View<bool *, host_t>("numNzWeight_", nWeightsPerRow_);
    for (size_t i = 0; i < numNzWeight_.extent(0); ++i) {
      numNzWeight_(i) = false;
    }
  }
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  void TpetraCrsMatrixAdapter<User,UserCoord>::setWeights(
    const scalar_t *weightVal, int stride, int idx)
{
  if (this->getPrimaryEntityType() == MATRIX_ROW)
    setRowWeights(weightVal, stride, idx);
  else {
    // TODO:  Need to allow weights for columns and/or nonzeros
    std::ostringstream emsg;
    emsg << __FILE__ << "," << __LINE__
         << " error:  setWeights not yet supported for"
         << " columns or nonzeros."
         << std::endl;
    throw std::runtime_error(emsg.str());
  }
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  void TpetraCrsMatrixAdapter<User,UserCoord>::setRowWeights(
    const scalar_t *weightVal, int stride, int idx)
{
  typedef StridedData<lno_t,scalar_t> input_t;
  if(idx<0 || idx >= nWeightsPerRow_)
  {
      std::ostringstream emsg;
      emsg << __FILE__ << ":" << __LINE__
           << "  Invalid row weight index " << idx << std::endl;
      throw std::runtime_error(emsg.str());
  }

  size_t nvtx = getLocalNumRows();
  ArrayRCP<const scalar_t> weightV(weightVal, 0, nvtx*stride, false);
  rowWeights_[idx] = input_t(weightV, stride);
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  void TpetraCrsMatrixAdapter<User,UserCoord>::setWeightIsDegree(
    int idx)
{
  if (this->getPrimaryEntityType() == MATRIX_ROW)
    setRowWeightIsNumberOfNonZeros(idx);
  else {
    // TODO:  Need to allow weights for columns and/or nonzeros
    std::ostringstream emsg;
    emsg << __FILE__ << "," << __LINE__
         << " error:  setWeightIsNumberOfNonZeros not yet supported for"
         << " columns" << std::endl;
    throw std::runtime_error(emsg.str());
  }
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  void TpetraCrsMatrixAdapter<User,UserCoord>::setRowWeightIsNumberOfNonZeros(
    int idx)
{
  if(idx<0 || idx >= nWeightsPerRow_)
  {
      std::ostringstream emsg;
      emsg << __FILE__ << ":" << __LINE__
           << "  Invalid row weight index " << idx << std::endl;
      throw std::runtime_error(emsg.str());
  }


  numNzWeight_[idx] = true;
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  template <typename Adapter>
    void TpetraCrsMatrixAdapter<User,UserCoord>::applyPartitioningSolution(
      const User &in, User *&out,
      const PartitioningSolution<Adapter> &solution) const
{
  // Get an import list (rows to be received)
  size_t numNewRows;
  ArrayRCP<gno_t> importList;
  try{
    numNewRows = Zoltan2::getImportList<Adapter,
                                        TpetraCrsMatrixAdapter<User,UserCoord> >
                                       (solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new matrix.
  RCP<User> outPtr = doMigration(in, numNewRows,importList.getRawPtr());
  out = const_cast<User *>(outPtr.get());
  outPtr.release();
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
  template <typename Adapter>
    void TpetraCrsMatrixAdapter<User,UserCoord>::applyPartitioningSolution(
      const User &in, RCP<User> &out,
      const PartitioningSolution<Adapter> &solution) const
{
  // Get an import list (rows to be received)
  size_t numNewRows;
  ArrayRCP<gno_t> importList;
  try{
    numNewRows = Zoltan2::getImportList<Adapter,
                                        TpetraCrsMatrixAdapter<User,UserCoord> >
                                       (solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new matrix.
  out = doMigration(in, numNewRows, importList.getRawPtr());
}

////////////////////////////////////////////////////////////////////////////
template <typename User, typename UserCoord>
RCP<User> TpetraCrsMatrixAdapter<User, UserCoord>::doMigration(
    const User &from, size_t numLocalRows, const gno_t *myNewRows) const {
  typedef Tpetra::Map<lno_t, gno_t, node_t> map_t;
  typedef Tpetra::CrsMatrix<scalar_t, lno_t, gno_t, node_t> tcrsmatrix_t;

  // We cannot create a Tpetra::RowMatrix, unless the underlying type is
  // something we know (like Tpetra::CrsMatrix).
  // If the underlying type is something different, the user probably doesn't
  // want a Tpetra::CrsMatrix back, so we throw an error.

  // Try to cast "from" matrix to a TPetra::CrsMatrix
  // If that fails we throw an error.
  // We could cast as a ref which will throw std::bad_cast but with ptr
  // approach it might be clearer what's going on here
  const tcrsmatrix_t *pCrsMatrix = dynamic_cast<const tcrsmatrix_t *>(&from);

  if (!pCrsMatrix) {
    throw std::logic_error("TpetraRowMatrixAdapter cannot migrate data for "
                           "your RowMatrix; it can migrate data only for "
                           "Tpetra::CrsMatrix.  "
                           "You can inherit from TpetraRowMatrixAdapter and "
                           "implement migration for your RowMatrix.");
  }

  // source map
  const RCP<const map_t> &smap = from.getRowMap();
  gno_t numGlobalRows = smap->getGlobalNumElements();
  gno_t base = smap->getMinAllGlobalIndex();

  // target map
  ArrayView<const gno_t> rowList(myNewRows, numLocalRows);
  const RCP<const Teuchos::Comm<int>> &comm = from.getComm();
  RCP<const map_t> tmap = rcp(new map_t(numGlobalRows, rowList, base, comm));

  // importer
  Tpetra::Import<lno_t, gno_t, node_t> importer(smap, tmap);

  // target matrix
  // Chris Siefert proposed using the following to make migration
  // more efficient.
  // By default, the Domain and Range maps are the same as in "from".
  // As in the original code, we instead set them both to tmap.
  // The assumption is a square matrix.
  // TODO:  what about rectangular matrices?
  // TODO:  Should choice of domain/range maps be an option to this function?

  // KDD 3/7/16:  disabling Chris' new code to avoid dashboard failures;
  // KDD 3/7/16:  can re-enable when issue #114 is fixed.
  // KDD 3/7/16:  when re-enable CSIEFERT code, can comment out
  // KDD 3/7/16:  "Original way" code.
  // CSIEFERT RCP<tcrsmatrix_t> M;
  // CSIEFERT from.importAndFillComplete(M, importer, tmap, tmap);

  // Original way we did it:

  int oldNumElts = smap->getLocalNumElements();
  int newNumElts = numLocalRows;

  // number of non zeros in my new rows
  typedef Tpetra::Vector<scalar_t, lno_t, gno_t, node_t> vector_t;
  vector_t numOld(smap); // TODO These vectors should have scalar=size_t,
  vector_t numNew(tmap); // but ETI does not yet support that.
  for (int lid = 0; lid < oldNumElts; lid++) {
    numOld.replaceGlobalValue(smap->getGlobalElement(lid),
                              scalar_t(from.getNumEntriesInLocalRow(lid)));
  }
  numNew.doImport(numOld, importer, Tpetra::INSERT);

  // TODO Could skip this copy if could declare vector with scalar=size_t.
  ArrayRCP<size_t> nnz(newNumElts);
  if (newNumElts > 0) {
    ArrayRCP<scalar_t> ptr = numNew.getDataNonConst(0);
    for (int lid = 0; lid < newNumElts; lid++) {
      nnz[lid] = static_cast<size_t>(ptr[lid]);
    }
  }

  RCP<tcrsmatrix_t> M = rcp(new tcrsmatrix_t(tmap, nnz()));

  M->doImport(from, importer, Tpetra::INSERT);
  M->fillComplete();

  // End of original way we did it.
  return Teuchos::rcp_dynamic_cast<User>(M);
}

}  //namespace Zoltan2

#endif