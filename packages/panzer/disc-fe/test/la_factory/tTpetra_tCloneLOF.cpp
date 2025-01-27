// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <string>
#include <iostream>

#include "PanzerDiscFE_config.hpp"
#include "Panzer_BlockedTpetraLinearObjFactory.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"
#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_LinearObjFactory_Utilities.hpp"

#include "Thyra_ProductVectorSpaceBase.hpp"
#include "Thyra_ProductVectorBase.hpp"
#include "Thyra_BlockedLinearOpBase.hpp"

#include "UnitTest_ConnManager.hpp"

#include "Teuchos_DefaultComm.hpp"

#include "Kokkos_DynRankView.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;

typedef Kokkos::DynRankView<double, PHX::Device> FieldContainer;

namespace panzer
{

   typedef double ScalarT;
   using LocalOrdinalT = panzer::LocalOrdinal;
   using GlobalOrdinalT = panzer::GlobalOrdinal;

   typedef panzer::TpetraLinearObjFactory<panzer::Traits, ScalarT, LocalOrdinalT, GlobalOrdinalT> TpetraLinObjFactoryType;
   typedef panzer::TpetraLinearObjContainer<ScalarT, LocalOrdinalT, GlobalOrdinalT> TpetraLinObjContainerType;
   typedef panzer::BlockedTpetraLinearObjFactory<panzer::Traits, ScalarT, LocalOrdinalT, GlobalOrdinalT> BlockedTpetraLinObjFactoryType;
   typedef panzer::BlockedTpetraLinearObjContainer<ScalarT, LocalOrdinalT, GlobalOrdinalT> BlockedTpetraLinObjContainerType;

   template <typename Intrepid2Type>
   RCP<const panzer::FieldPattern> buildFieldPattern()
   {
      // build a geometric pattern from a single basis
      RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = rcp(new Intrepid2Type);
      RCP<const panzer::FieldPattern> pattern = rcp(new panzer::Intrepid2FieldPattern(basis));
      return pattern;
   }

   TEUCHOS_UNIT_TEST(tCloneLOF, tpetra)
   {
      // build global (or serial communicator)
#ifdef HAVE_MPI
      Teuchos::RCP<const Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
      Teuchos::RCP<const Teuchos::SerialComm<int>> tComm = Teuchos::rcp(new Teuchos::SerialComm<int>(MPI_COMM_WORLD));
#endif

      int myRank = tComm->getRank();
      int numProc = tComm->getSize();

      RCP<ConnManager> connManager = rcp(new unit_test::ConnManager(myRank, numProc));

      RCP<const FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();

      RCP<panzer::DOFManager> indexer = rcp(new panzer::DOFManager());
      indexer->setConnManager(connManager, MPI_COMM_WORLD);
      indexer->addField("U", patternC1);
      indexer->addField("V", patternC1);
      indexer->buildGlobalUnknowns();

      RCP<panzer::DOFManager> control_indexer = rcp(new panzer::DOFManager());
      control_indexer->setConnManager(connManager, MPI_COMM_WORLD);
      control_indexer->addField("Z", patternC1);
      control_indexer->buildGlobalUnknowns();

      // setup factory
      RCP<TpetraLinObjFactoryType> t_lof = Teuchos::rcp(new TpetraLinObjFactoryType(tComm.getConst(), indexer));

      // this is the member we are testing!
      RCP<const LinearObjFactory<Traits>> control_lof = cloneWithNewDomain(*t_lof, control_indexer);
      RCP<const TpetraLinObjFactoryType> t_control_lof = rcp_dynamic_cast<const TpetraLinObjFactoryType>(control_lof);

      std::vector<panzer::GlobalOrdinal> control_owned;
      control_indexer->getOwnedIndices(control_owned);

      TEST_ASSERT(t_control_lof->getMap()->isSameAs(*t_lof->getMap()));
      TEST_EQUALITY(t_control_lof->getColMap()->getLocalNumElements(), control_owned.size());
   }

   TEUCHOS_UNIT_TEST(tCloneLOF, blocked_tpetra)
   {
      typedef Thyra::ProductVectorBase<double> PVector;
      typedef Thyra::BlockedLinearOpBase<double> BLinearOp;

// build global (or serial communicator)
#ifdef HAVE_MPI
      Teuchos::RCP<const Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
      Teuchos::RCP<const Teuchos::SerialComm<int>> tComm = Teuchos::rcp(new Teuchos::SerialComm<int>(MPI_COMM_WORLD));
#endif

      int myRank = tComm->getRank();
      int numProc = tComm->getSize();

      RCP<ConnManager> connManager = rcp(new unit_test::ConnManager(myRank, numProc));

      RCP<const FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();
      RCP<const FieldPattern> patternC2 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::exec_space, double, double>>();

      RCP<panzer::BlockedDOFManager> indexer = rcp(new panzer::BlockedDOFManager());
      {
         std::vector<std::vector<std::string> > fieldOrder(2);

         indexer->setConnManager(connManager, MPI_COMM_WORLD);
         indexer->addField("U", patternC1);
         indexer->addField("V", patternC1);

         fieldOrder[0].push_back("U");
         fieldOrder[1].push_back("V");
         indexer->setFieldOrder(fieldOrder);
         indexer->buildGlobalUnknowns();
      }

      RCP<panzer::BlockedDOFManager> control_indexer = rcp(new panzer::BlockedDOFManager());
      {
         std::vector<std::vector<std::string>> fieldOrder(1);
         control_indexer->setConnManager(connManager, MPI_COMM_WORLD);
         control_indexer->addField("Z", patternC1);
         fieldOrder[0].push_back("Z");
         indexer->setFieldOrder(fieldOrder);
         control_indexer->buildGlobalUnknowns();

         patternC2->print(out);

         {
            std::vector<panzer::GlobalOrdinal> gids;
            control_indexer->getFieldDOFManagers()[0]->getOwnedIndices(gids);
            out << "GIDs 0 = " << gids.size() << std::endl;
         }
      }

      // setup factory
      RCP<BlockedTpetraLinObjFactoryType> bt_lof = Teuchos::rcp(new BlockedTpetraLinObjFactoryType(tComm, indexer));
      RCP<const LinearObjFactory<Traits>> control_lof = cloneWithNewDomain(*bt_lof, control_indexer);

      TEST_ASSERT(nullptr != control_lof.getRawPtr());

      RCP<const BlockedTpetraLinObjFactoryType> bt_control_lof = rcp_dynamic_cast<const BlockedTpetraLinObjFactoryType>(control_lof, true);

      RCP<BLinearOp> mat = rcp_dynamic_cast<BLinearOp>(bt_control_lof->getThyraMatrix());
      RCP<BLinearOp> gmat = rcp_dynamic_cast<BLinearOp>(bt_control_lof->getGhostedThyraMatrix());
      RCP<PVector> x = rcp_dynamic_cast<PVector>(bt_control_lof->getThyraDomainVector());
      RCP<PVector> gx = rcp_dynamic_cast<PVector>(bt_control_lof->getGhostedThyraDomainVector());
      RCP<PVector> f = rcp_dynamic_cast<PVector>(bt_control_lof->getThyraRangeVector());
      RCP<PVector> gf = rcp_dynamic_cast<PVector>(bt_control_lof->getGhostedThyraRangeVector());

      TEST_EQUALITY(x->productSpace()->numBlocks(), 1);
      TEST_EQUALITY(x->productSpace()->dim(), 18);
      TEST_EQUALITY(gx->productSpace()->numBlocks(), 1);
      TEST_EQUALITY(gx->productSpace()->dim(), 10 + 15);

      TEST_EQUALITY(f->productSpace()->numBlocks(), 2);
      TEST_EQUALITY(f->productSpace()->dim(), 36);
      TEST_EQUALITY(gf->productSpace()->numBlocks(), 2);
      TEST_EQUALITY(gf->productSpace()->dim(), 50);

      TEST_EQUALITY(mat->productRange()->numBlocks(), 2);
      TEST_EQUALITY(mat->productRange()->dim(), 36);
      TEST_EQUALITY(mat->productDomain()->numBlocks(), 1);
      TEST_EQUALITY(mat->productDomain()->dim(), 18);

      TEST_EQUALITY(gmat->productRange()->numBlocks(), 2);
      TEST_EQUALITY(gmat->productRange()->dim(), 50);
      TEST_EQUALITY(gmat->productDomain()->numBlocks(), 1);
      TEST_EQUALITY(gmat->productDomain()->dim(), 10 + 15);
   }

   TEUCHOS_UNIT_TEST(tCloneLOF, blocked_tpetra_nonblocked_domain)
   {
      typedef Thyra::ProductVectorBase<double> PVector;
      typedef Thyra::BlockedLinearOpBase<double> BLinearOp;
      typedef Thyra::VectorBase<double> Vector;

// build global (or serial communicator)
#ifdef HAVE_MPI
      Teuchos::RCP<const Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
      Teuchos::RCP<const Teuchos::SerialComm<int>> tComm = Teuchos::rcp(new Teuchos::SerialComm<int>(MPI_COMM_WORLD));
#endif

      int myRank = tComm->getRank();
      int numProc = tComm->getSize();

      RCP<ConnManager> connManager = rcp(new unit_test::ConnManager(myRank, numProc));

      RCP<const FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();
      RCP<const FieldPattern> patternC2 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::exec_space, double, double>>();

      RCP<panzer::BlockedDOFManager> indexer = rcp(new panzer::BlockedDOFManager());
      {
         std::vector<std::vector<std::string>> fieldOrder(2);

         indexer->setConnManager(connManager, MPI_COMM_WORLD);
         indexer->addField("U", patternC1);
         indexer->addField("V", patternC1);

         fieldOrder[0].push_back("U");
         fieldOrder[1].push_back("V");
         indexer->setFieldOrder(fieldOrder);
         indexer->buildGlobalUnknowns();
      }

      RCP<panzer::DOFManager> control_indexer = rcp(new panzer::DOFManager());
      {
         control_indexer->setConnManager(connManager, MPI_COMM_WORLD);
         control_indexer->addField("Z", patternC1);
         control_indexer->buildGlobalUnknowns();

         patternC2->print(out);
      }

      // setup factory
      RCP<BlockedTpetraLinObjFactoryType> bt_lof = Teuchos::rcp(new BlockedTpetraLinObjFactoryType(tComm, indexer));

      // this is the member we are testing!
      RCP<const LinearObjFactory<Traits>> control_lof = cloneWithNewDomain(*bt_lof, control_indexer);
      TEST_ASSERT(nullptr != control_lof.getRawPtr());

      RCP<const BlockedTpetraLinObjFactoryType> bt_control_lof = rcp_dynamic_cast<const BlockedTpetraLinObjFactoryType>(control_lof, true);
      TEST_ASSERT(nullptr != bt_control_lof.getRawPtr());

      RCP<BLinearOp> mat = rcp_dynamic_cast<BLinearOp>(bt_control_lof->getThyraMatrix(), true);
      RCP<BLinearOp> gmat = rcp_dynamic_cast<BLinearOp>(bt_control_lof->getGhostedThyraMatrix(), true);
      RCP<Vector> x = bt_control_lof->getThyraDomainVector();
      RCP<Vector> gx = bt_control_lof->getGhostedThyraDomainVector();
      RCP<PVector> f = rcp_dynamic_cast<PVector>(bt_control_lof->getThyraRangeVector(), true);
      RCP<PVector> gf = rcp_dynamic_cast<PVector>(bt_control_lof->getGhostedThyraRangeVector(), true);

      TEST_EQUALITY(x->space()->dim(), 18);
      TEST_EQUALITY(gx->space()->dim(), 10 + 15);

      TEST_EQUALITY(f->productSpace()->numBlocks(), 2);
      TEST_EQUALITY(f->productSpace()->dim(), 36);
      TEST_EQUALITY(gf->productSpace()->numBlocks(), 2);
      TEST_EQUALITY(gf->productSpace()->dim(), 50);

      TEST_EQUALITY(mat->productRange()->numBlocks(), 2);
      TEST_EQUALITY(mat->productRange()->dim(), 36);
      TEST_EQUALITY(mat->productDomain()->numBlocks(), 1);
      TEST_EQUALITY(mat->productDomain()->dim(), 18);

      TEST_EQUALITY(gmat->productRange()->numBlocks(), 2);
      TEST_EQUALITY(gmat->productRange()->dim(), 50);
      TEST_EQUALITY(gmat->productDomain()->numBlocks(), 1);
      TEST_EQUALITY(gmat->productDomain()->dim(), 10 + 15);
   }

}
