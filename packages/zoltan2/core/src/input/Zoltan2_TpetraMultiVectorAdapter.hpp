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

/*! \file Zoltan2_TpetraMultiVectorAdapter.hpp
    \brief Defines the TpetraMultiVectorAdapter
*/

#ifndef _ZOLTAN2_TPETRAMULTIVECTORADAPTER_HPP_
#define _ZOLTAN2_TPETRAMULTIVECTORADAPTER_HPP_

#include <Zoltan2_VectorAdapter.hpp>
#include <Zoltan2_StridedData.hpp>
#include <Zoltan2_PartitioningHelpers.hpp>

namespace Zoltan2 {

/*!  \brief An adapter for Tpetra::MultiVector.

    The template parameter is the user's input object:
    \li \c Tpetra::MultiVector

    The \c scalar_t type, representing use data such as vector values, is
    used by Zoltan2 for weights, coordinates, part sizes and
    quality metrics.
    Some User types (like Tpetra::CrsMatrix) have an inherent scalar type,
    and some
    (like Tpetra::CrsGraph) do not.  For such objects, the scalar type is
    set by Zoltan2 to \c float.  If you wish to change it to double, set
    the second template parameter to \c double.
*/

template <typename User>
  class TpetraMultiVectorAdapter : public VectorAdapter<User> {
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef typename InputTraits<User>::scalar_t scalar_t;
  typedef typename InputTraits<User>::lno_t    lno_t;
  typedef typename InputTraits<User>::gno_t    gno_t;
  typedef typename InputTraits<User>::part_t   part_t;
  typedef typename InputTraits<User>::node_t   node_t;
  typedef User user_t;
  typedef User userCoord_t;

  typedef Tpetra::MultiVector<scalar_t, lno_t, gno_t, node_t> t_mvector_t;
#endif

  /*! \brief Constructor
   *
   *  \param invector  the user's Tpetra MultiVector object
   *  \param weights  a list of pointers to arrays of weights.
   *      The number of weights per multivector element is assumed to be
   *      \c weights.size().
   *  \param weightStrides  a list of strides for the \c weights.
   *     The weight for weight index \c n for multivector element
   *     \c k should be found at <tt>weights[n][weightStrides[n] * k]</tt>.
   *     If \c weightStrides.size() is zero, it is assumed all strides are one.
   *
   *  The values pointed to the arguments must remain valid for the
   *  lifetime of this Adapter.
   */

  TpetraMultiVectorAdapter(const RCP<const User> &invector,
    std::vector<const scalar_t *> &weights, std::vector<int> &weightStrides);

  /*! \brief Constructor for case when weights are not being used.
   *
   *  \param invector  the user's Tpetra MultiVector object
   */

  TpetraMultiVectorAdapter(const RCP<const User> &invector);


  ////////////////////////////////////////////////////
  // The Adapter interface.
  ////////////////////////////////////////////////////

  size_t getLocalNumIDs() const { return vector_->getLocalLength();}

  void getIDsView(const gno_t *&ids) const
  {
    ids = map_->getLocalElementList().getRawPtr();
  }

  void getIDsKokkosView(
    Kokkos::View<const gno_t *, typename node_t::device_type> &ids) const {
      using device_type = typename node_t::device_type;
      // MJ can be running Host, CudaSpace, or CudaUVMSpace while Map now
      // internally never stores CudaUVMSpace so we may need a conversion.
      // However Map stores both Host and CudaSpace so this could be improved
      // if device_type was CudaSpace. Then we could add a new accessor to
      // Map such as getMyGlobalIndicesDevice() which could be direct assigned
      // here. Since Tpetra is still UVM dependent that is not going to happen
      // yet so just leaving this as Host to device_type conversion for now.
      ids = Kokkos::create_mirror_view_and_copy(device_type(),
        map_->getMyGlobalIndices());
  }

  int getNumWeightsPerID() const { return numWeights_;}

  void getWeightsView(const scalar_t *&weights, int &stride, int idx) const
  {
    if(idx<0 || idx >= numWeights_)
    {
        std::ostringstream emsg;
        emsg << __FILE__ << ":" << __LINE__
             << "  Invalid weight index " << idx << std::endl;
        throw std::runtime_error(emsg.str());
    }

    size_t length;
    weights_[idx].getStridedList(length, weights, stride);
  }

  void getWeightsKokkos2dView(Kokkos::View<scalar_t **,
    typename node_t::device_type> &wgt) const {
    typedef Kokkos::View<scalar_t**, typename node_t::device_type> view_t;
    wgt = view_t("wgts", vector_->getLocalLength(), numWeights_);
    typename view_t::HostMirror host_wgt = Kokkos::create_mirror_view(wgt);
    for(int idx = 0; idx < numWeights_; ++idx) {
      const scalar_t * weights;
      size_t length;
      int stride;
      weights_[idx].getStridedList(length, weights, stride);
      size_t fill_index = 0;
      for(size_t n = 0; n < length; n += stride) {
        host_wgt(fill_index++,idx) = weights[n];
      }
    }
    Kokkos::deep_copy(wgt, host_wgt);
  }

  ////////////////////////////////////////////////////
  // The VectorAdapter interface.
  ////////////////////////////////////////////////////

  int getNumEntriesPerID() const {return vector_->getNumVectors();}

  void getEntriesView(const scalar_t *&elements, int &stride, int idx=0) const;

  void getEntriesKokkosView(
    // coordinates in MJ are LayoutLeft since Tpetra Multivector gives LayoutLeft
    Kokkos::View<scalar_t **, Kokkos::LayoutLeft,
    typename node_t::device_type> & elements) const;

  template <typename Adapter>
    void applyPartitioningSolution(const User &in, User *&out,
         const PartitioningSolution<Adapter> &solution) const;

  template <typename Adapter>
    void applyPartitioningSolution(const User &in, RCP<User> &out,
         const PartitioningSolution<Adapter> &solution) const;

private:

  // MPL: 07/20/2023: TOCHECK: invector_ seems to be useless
  RCP<const User> invector_;
  RCP<const t_mvector_t> vector_;
  RCP<const Tpetra::Map<lno_t, gno_t, node_t> > map_;

  int numWeights_;
  ArrayRCP<StridedData<lno_t, scalar_t> > weights_;

  RCP<User> doMigration(const User &from, size_t numLocalRows,
                        const gno_t *myNewRows) const;
};

////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////

template <typename User>
  TpetraMultiVectorAdapter<User>::TpetraMultiVectorAdapter(
    const RCP<const User> &invector,
    std::vector<const scalar_t *> &weights, std::vector<int> &weightStrides):
      invector_(invector), vector_(), map_(),
      numWeights_(weights.size()), weights_(weights.size())
{
  typedef StridedData<lno_t, scalar_t> input_t;
  // MPL: 07/13/86: should we copy the data from invector to vector_ ?
  vector_ = invector;

  map_ = vector_->getMap();

  size_t length = vector_->getLocalLength();

  if (length > 0 && numWeights_ > 0){
    int stride = 1;
    for (int w=0; w < numWeights_; w++){
      if (weightStrides.size())
        stride = weightStrides[w];
      ArrayRCP<const scalar_t> wgtV(weights[w], 0, stride*length, false);
      weights_[w] = input_t(wgtV, stride);
    }
  }
}


////////////////////////////////////////////////////////////////////////////
template <typename User>
  TpetraMultiVectorAdapter<User>::TpetraMultiVectorAdapter(
    const RCP<const User> &invector):
      invector_(invector), vector_(), map_(),
      numWeights_(0), weights_()
{
  // MPL: 07/13/86: should we copy the data from invector to vector_ ?
  vector_ = invector;
  map_ = vector_->getMap();
}

////////////////////////////////////////////////////////////////////////////
template <typename User>
  void TpetraMultiVectorAdapter<User>::getEntriesView(
    const scalar_t *&elements, int &stride, int idx) const
{
  size_t vecsize;
  stride = 1;
  elements = NULL;

  vecsize = vector_->getLocalLength();
  if (vecsize > 0){
    ArrayRCP<const scalar_t> data = vector_->getData(idx);
    elements = data.get();
  }
}

////////////////////////////////////////////////////////////////////////////
template <typename User>
  void TpetraMultiVectorAdapter<User>::getEntriesKokkosView(
    // coordinates in MJ are LayoutLeft since Tpetra Multivector gives LayoutLeft
    Kokkos::View<scalar_t **, Kokkos::LayoutLeft, typename node_t::device_type> & elements) const
{
    // coordinates in MJ are LayoutLeft since Tpetra Multivector gives LayoutLeft
       Tpetra::MultiVector<scalar_t, lno_t, gno_t, node_t> vec = *vector_.get();
    Kokkos::View<scalar_t **, Kokkos::LayoutLeft, typename node_t::device_type> view2d =
      vec.template getLocalView<typename node_t::device_type>(Tpetra::Access::ReadWrite);
    elements = view2d;

}

////////////////////////////////////////////////////////////////////////////
template <typename User>
  template <typename Adapter>
    void TpetraMultiVectorAdapter<User>::applyPartitioningSolution(
      const User &in, User *&out,
      const PartitioningSolution<Adapter> &solution) const
{
  // Get an import list (rows to be received)
  size_t numNewRows;
  ArrayRCP<gno_t> importList;
  try{
    numNewRows = Zoltan2::getImportList<Adapter,
                                        TpetraMultiVectorAdapter<User> >
                                       (solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new vector.
  RCP<User> outPtr = doMigration(in, numNewRows, importList.getRawPtr());
  out = outPtr.get();
  outPtr.release();
}

////////////////////////////////////////////////////////////////////////////
template <typename User>
  template <typename Adapter>
    void TpetraMultiVectorAdapter<User>::applyPartitioningSolution(
      const User &in, RCP<User> &out,
      const PartitioningSolution<Adapter> &solution) const
{
  // Get an import list (rows to be received)
  size_t numNewRows;
  ArrayRCP<gno_t> importList;
  try{
    numNewRows = Zoltan2::getImportList<Adapter,
                                        TpetraMultiVectorAdapter<User> >
                                       (solution, this, importList);
  }
  Z2_FORWARD_EXCEPTIONS;

  // Move the rows, creating a new vector.
  out = doMigration(in, numNewRows, importList.getRawPtr());
}

////////////////////////////////////////////////////////////////////////////
template < typename User>
RCP<User> TpetraMultiVectorAdapter<User>::doMigration(
  const User &from,
  size_t numLocalRows,
  const gno_t *myNewRows
) const
{
  typedef Tpetra::Map<lno_t, gno_t, node_t> map_t;

  // source map
  const RCP<const map_t> &smap = from.getMap();
  gno_t numGlobalElts = smap->getGlobalNumElements();
  gno_t base = smap->getMinAllGlobalIndex();

  // target map
  ArrayView<const gno_t> eltList(myNewRows, numLocalRows);
  const RCP<const Teuchos::Comm<int> > comm = from.getMap()->getComm();
  RCP<const map_t> tmap = rcp(new map_t(numGlobalElts, eltList, base, comm));

  // importer
  Tpetra::Import<lno_t, gno_t, node_t> importer(smap, tmap);

  // target vector
  RCP<t_mvector_t> MV = rcp(
    new t_mvector_t(tmap, from.getNumVectors(), true));
  MV->doImport(from, importer, Tpetra::INSERT);

  gno_t base2 = smap->getMinAllGlobalIndex();

  return MV;
}

}  //namespace Zoltan2

#endif
