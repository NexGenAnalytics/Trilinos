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

/*! \file Zoltan2_BasicIdentifierAdapter.hpp
    \brief Defines the BasicIdentifierAdapter class.
*/

#ifndef _ZOLTAN2_BASICIDENTIFIERADAPTER_HPP_
#define _ZOLTAN2_BASICIDENTIFIERADAPTER_HPP_

#include <Kokkos_Core.hpp>
#include <Zoltan2_IdentifierAdapter.hpp>
#include <Zoltan2_StridedData.hpp>

namespace Zoltan2 {

/*! \brief This class represents a collection of global Identifiers
 *           and their associated weights, if any.
 *
 *  The user supplies the identifiers and weights by way of pointers
 *    to arrays.
 *
 *  The template parameter (\c User) is a C++ class type which provides the
 *  actual data types with which the Zoltan2 library will be compiled, through
 *  a Traits mechanism.  \c User may be the
 *  actual class used by application to represent coordinates, or it may be
 *  the empty helper class \c BasicUserTypes with which a Zoltan2 user
 *  can easily supply the data types for the library.
 *
 *  The \c scalar_t type, representing use data such as matrix values, is
 *  used by Zoltan2 for weights, coordinates, part sizes and
 *  quality metrics.
 *  Some User types (like Tpetra::CrsMatrix) have an inherent scalar type,
 *  and some
 *  (like Tpetra::CrsGraph) do not.  For such objects, the scalar type is
 *  set by Zoltan2 to \c float.  If you wish to change it to double, set
 *  the second template parameter to \c double.
 */

template <typename User>
  class BasicIdentifierAdapter: public IdentifierAdapter<User> {

public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  using scalar_t = typename InputTraits<User>::scalar_t;
  using lno_t    = typename InputTraits<User>::lno_t;
  using gno_t    = typename InputTraits<User>::gno_t;
  using part_t   = typename InputTraits<User>::part_t;
  using node_t   = typename InputTraits<User>::node_t;
  using device_t = typename node_t::device_type;
  using user_t   = User;

  using Base = IdentifierAdapter<User>;
#endif

  /*! \brief Constructor
   *  \param numIds is the number of identifiers in the list
   *  \param ids should point to a list of numIds identifiers.
   *  \param weights  a list of pointers to arrays of weights.
   *      The number of weights per identifier is assumed to be
   *      \c weights.size().
   *  \param weightStrides  a list of strides for the \c weights.
   *     The weight for weight index \c n for \c ids[k] should be
   *     found at <tt>weights[n][weightStrides[n] * k]</tt>.
   *     If \c weightStrides.size() is zero, it is assumed all strides are one.
   *
   *  The values pointed to the arguments must remain valid for the
   *  lifetime of this Adapter.
   */
  BasicIdentifierAdapter(lno_t numIds, const gno_t *idPtr,
    std::vector<const scalar_t *> &weights, std::vector<int> &weightStrides);

  /*! \brief Constructor
   *  \param numIds is the number of identifiers in the list
   *  \param ids should point to a list of numIds identifiers.
   *
   *  The values pointed to the arguments must remain valid for the
   *  lifetime of this Adapter.
   */
  BasicIdentifierAdapter(lno_t numIds, const gno_t *idPtr):
      localNumIDs_(numIds), idList_(idPtr), weights_() {}

  /*! \brief Constructor
   *  \param ids should point to a View of identifiers.
   *  \param weights  a list of pointers to arrays of weights.
   *      The number of weights per identifier is assumed to be
   *      \c weights.extent(1).
   *  \param weightStrides  a list of strides for the \c weights.
   *     The weight for weight index \c n for \c ids[k] should be
   *     found at <tt>weights[n][weightStrides[n] * k]</tt>.
   *     If \c weightStrides.size() is zero, it is assumed all strides are one.
   *
   *  The values pointed to the arguments must remain valid for the
   *  lifetime of this Adapter.
   */
  BasicIdentifierAdapter(
    Kokkos::View<gno_t*, device_t> &ids,
    Kokkos::View<scalar_t**, device_t> &weights);

  ////////////////////////////////////////////////////////////////
  // The Adapter interface.
  ////////////////////////////////////////////////////////////////

  size_t getLocalNumIDs() const {
      return localNumIDs_;
  }

  void getIDsView(const gno_t *&ids) const {
      ids = idList_;
  }

  void getIDsKokkosView(Kokkos::View<const gno_t *, device_t> &ids) const override {
    ids = idsView_;
  }

  void getIDsDeviceView(
      typename Base::ConstIdsDeviceView &ids) const {
    ids = idsView_;
  }

  void getIDsHostView(
      typename Base::ConstIdsHostView &ids) const {
    auto hostIds = Kokkos::create_mirror_view(idsView_);
    Kokkos::deep_copy(hostIds, idsView_);
    ids = hostIds;
  }

  int getNumWeightsPerID() const {
    return numWeightsPerID_;
  }

  void getWeightsView(const scalar_t *&wgt, int &stride,
                      int idx = 0) const {
    if (idx < 0 || idx >= weights_.size()) {
      std::ostringstream emsg;
      emsg << __FILE__ << ":" << __LINE__
          << "  Invalid weight index " << idx << std::endl;
      throw std::runtime_error(emsg.str());
    }
    size_t length;
    weights_[idx].getStridedList(length, wgt, stride);
  }

  void getWeightsKokkosView(Kokkos::View<scalar_t **, device_t> &wgts) const override {
    wgts = weightsView_;
  }

  void getWeightsDeviceView(typename Base::WeightsDeviceView &wgts) const override {
    wgts = weightsView_; // weightsView_ is already on device
  }

  void getWeightsHostView(typename Base::WeightsHostView &wgts) const override {
    auto hostWgts = Kokkos::create_mirror_view(weightsView_);
    Kokkos::deep_copy(hostWgts, weightsView_);
    wgts = hostWgts;
  }

private:
  lno_t localNumIDs_;
  const gno_t *idList_;
  ArrayRCP<StridedData<lno_t, scalar_t> > weights_;
  size_t numWeightsPerID_ = 0;

  Kokkos::View<gno_t *, device_t> idsView_;
  Kokkos::View<scalar_t **, device_t> weightsView_;
};

////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////

template <typename User>
  BasicIdentifierAdapter<User>::BasicIdentifierAdapter(
    lno_t numIds, const gno_t *idPtr,
    std::vector<const scalar_t *> &weights, std::vector<int> &weightStrides):
      localNumIDs_(numIds), idList_(idPtr), weights_() {
  typedef StridedData<lno_t, scalar_t> input_t;
  size_t numWeights = weights.size();
  numWeightsPerID_ = numWeights;

  if (numWeights > 0){
    weights_ = arcp(new input_t [numWeights], 0, numWeights, true);

    if (numIds > 0){
      for (size_t i = 0; i < numWeights; i++){
        int stride = weightStrides.size() ? weightStrides[i] : 1;
        ArrayRCP<const scalar_t> wgtV(weights[i], 0, stride * numIds, false);
        weights_[i] = input_t(wgtV, stride);
      }
    }
  }
}

template <typename User>
BasicIdentifierAdapter<User>::BasicIdentifierAdapter(
    Kokkos::View<gno_t *, device_t> &ids,
    Kokkos::View<scalar_t **, device_t> &weights) {
  idsView_ = Kokkos::View<gno_t *, device_t>("idsView_", ids.extent(0));
  Kokkos::deep_copy(idsView_, ids);

  weightsView_ = Kokkos::View<scalar_t **, device_t>("weightsView_",
                                                         weights.extent(0),
                                                         weights.extent(1));
  Kokkos::deep_copy(weightsView_, weights);
  localNumIDs_ = idsView_.extent(0);
  numWeightsPerID_ = weights.extent(1);
}

}  //namespace Zoltan2

#endif
