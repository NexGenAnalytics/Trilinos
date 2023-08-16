/*
// @HEADER
//
// ***********************************************************************
//
//      Teko: A package for block and physics based preconditioning
//                  Copyright 2010 Sandia Corporation
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
// Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

*/

#include <string>
#include <iostream>

// Teuchos includes
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

// Teko-Package includes
#include "Teko_Config.h"
#include "Teko_Utilities.hpp"
#include "Teko_DiagonallyScaledPreconditionerFactory.hpp"
#include "Teko_PreconditionerInverseFactory.hpp"
#include "Teko_PreconditionerLinearOp.hpp"
#include "Teko_ProbingPreconditionerFactory.hpp"
#include "Teko_InverseLibrary.hpp"

// Thyra testing tools
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_LinearOpTester.hpp"

// Tpetra includes
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>

#ifdef Teko_ENABLE_Isorropia

// Teuchos using
using Teuchos::rcp;
using Teuchos::RCP;

// Tpetra using
using ST = typename Tpetra::Vector<double>::scalar_type;
using LO = typename Tpetra::Vector<>::local_ordinal_type;
using GO = typename Tpetra::Vector<>::global_ordinal_type;
using NT = typename Tpetra::Vector<>::node_type;
using tmap_t = Tpetra::Map<LO,GO,NT>;
using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;

const RCP<Thyra::LinearOpBase<ST>> buildSystem(const RCP<const Teuchos::Comm<int>> comm, int size)
{
  tmap_t map(size, 0, comm);
  RCP<tcrsmatrix_t> mat = rcp(new tcrsmatrix_t(map, 0));
   
  double values[] = { -1.0, 2.0, -1.0};
  int iTemp[] = {-1,0,1}, indices[3];
  double * vPtr;
  int * iPtr;

  for(int i=0; i<map.getLocalNumElements(); i++) {
    int count = 3;
    int gid = map.getIndexBase(i);

    vPtr = values;
    iPtr = indices;

    indices[0] = gid+iTemp[0];
    indices[1] = gid+iTemp[1];
    indices[2] = gid+iTemp[2];

    if(gid==0) {
      vPtr = &values[1];
      iPtr = &indices[1];
      count = 2;
    }
    else if(gid==map.getMaxGlobalIndex()) {
      count = 2;
      mat->insertGlobalValues(gid, count, vPtr, iPtr);
    }
  }

  mat->fillComplete();

  const RCP<const VectorSpaceBase<ST> > rangeSpace =
    Thyra::createVectorSpace<ST>(mat->getRangeMap());
  const RCP<const VectorSpaceBase<ST> > domainSpace =
    Thyra::createVectorSpace<ST>(mat->getDomainMap());

  return Thyra::tpetraLinearOp(rangeSpace, domainSpace, mat);
}

TEUCHOS_UNIT_TEST(tProbingFactory_Tpetra, basic_test)
{
  // build global (or serial communicator)
  auto Comm = Tpetra::getDefaultComm();
  Teko::LinearOp lo = buildSystem(Comm, 10);

  RCP<Teko::InverseLibrary> invLib = Teko::InverseLibrary::buildFromStratimikos();
  RCP<Teko::InverseFactory> directSolveFactory = invLib->getInverseFactory("Amesos");

  RCP<Teko::ProbingPreconditionerFactory> probeFact =
    rcp(new Teko::ProbingPreconditionerFactory);
  probeFact->setGraphOperator(lo);
  probeFact->setInverseFactory(directSolveFactory);

  RCP<Teko::InverseFactory> invFact =
    rcp(new Teko::PreconditionerInverseFactory(probeFact,Teuchos::null));

  Teko::LinearOp probedInverse = Teko::buildInverse(*invFact,lo);
  Teko::LinearOp invLo = Teko::buildInverse(*directSolveFactory,lo);

  Thyra::LinearOpTester<double> tester;
  tester.dump_all(true);
  tester.show_all_tests(true);
  
  {
    const bool result = tester.compare( *probedInverse, *invLo, Teuchos::ptrFromRef(out));
    if (!result) {
      out << "Apply: FAILURE" << std::endl;
      success = false;
    }
    else
      out << "Apply: SUCCESS" << std::endl;
  }
}

/*TEUCHOS_UNIT_TEST(tProbingFactory, parameterlist_constr)
{
   // build global (or serial communicator)
   #ifdef HAVE_MPI
      Epetra_MpiComm Comm(MPI_COMM_WORLD);
   #else
      Epetra_SerialComm Comm;
   #endif

   Teko::LinearOp lo = buildSystem(Comm,10);

   Teuchos::RCP<Teko::InverseLibrary> invLib = Teko::InverseLibrary::buildFromStratimikos();
   Teuchos::RCP<Teko::InverseFactory> directSolveFactory = invLib->getInverseFactory("Amesos");

   {
      Teuchos::ParameterList pl;
      pl.set("Inverse Type","Amesos");
      pl.set("Probing Graph Operator",lo);

      Teuchos::RCP<Teko::ProbingPreconditionerFactory> probeFact
            = rcp(new Teko::ProbingPreconditionerFactory);
      probeFact->initializeFromParameterList(pl);

      RCP<Teko::InverseFactory> invFact = Teuchos::rcp(new Teko::PreconditionerInverseFactory(probeFact,Teuchos::null));

      Teko::LinearOp probedInverse = Teko::buildInverse(*invFact,lo);
      Teko::LinearOp invLo = Teko::buildInverse(*directSolveFactory,lo);

      Thyra::LinearOpTester<double> tester;
      tester.dump_all(true);
      tester.show_all_tests(true);

      {
         const bool result = tester.compare( *probedInverse, *invLo, Teuchos::ptrFromRef(out));
         if (!result) {
            out << "Apply: FAILURE" << std::endl;
            success = false;
         }
         else
            out << "Apply: SUCCESS" << std::endl;
      }
   }

   {
      Teuchos::RCP<const Epetra_CrsGraph> theGraph
            = rcpFromRef(rcp_dynamic_cast<const Epetra_CrsMatrix>(Thyra::get_Epetra_Operator(*lo))->Graph());

      Teuchos::ParameterList pl;
      pl.set("Inverse Type","Amesos");
      pl.set("Probing Graph",theGraph);

      Teuchos::RCP<Teko::ProbingPreconditionerFactory> probeFact
            = rcp(new Teko::ProbingPreconditionerFactory);
      probeFact->initializeFromParameterList(pl);

      RCP<Teko::InverseFactory> invFact = Teuchos::rcp(new Teko::PreconditionerInverseFactory(probeFact,Teuchos::null));

      Teko::LinearOp probedInverse = Teko::buildInverse(*invFact,lo);
      Teko::LinearOp invLo = Teko::buildInverse(*directSolveFactory,lo);

      Thyra::LinearOpTester<double> tester;
      tester.dump_all(true);
      tester.show_all_tests(true);

      {
         const bool result = tester.compare( *probedInverse, *invLo, Teuchos::ptrFromRef(out));
         if (!result) {
            out << "Apply: FAILURE" << std::endl;
            success = false;
         }
         else
            out << "Apply: SUCCESS" << std::endl;
      }
   }
}

TEUCHOS_UNIT_TEST(tProbingFactory, invlib_constr)
{
   // build global (or serial communicator)
   #ifdef HAVE_MPI
      Epetra_MpiComm Comm(MPI_COMM_WORLD);
   #else
      Epetra_SerialComm Comm;
   #endif

   Teko::LinearOp lo = buildSystem(Comm,10);

   Teuchos::ParameterList subList;
   subList.set("Type","Probing Preconditioner");
   subList.set("Inverse Type","Amesos");
   subList.set("Probing Graph Operator",lo);

   Teuchos::ParameterList pl;
   pl.set("Prober",subList);


   Teuchos::RCP<Teko::InverseLibrary> invLib = Teko::InverseLibrary::buildFromParameterList(pl);
   Teuchos::RCP<Teko::InverseFactory> proberFactory = invLib->getInverseFactory("Prober");
   Teuchos::RCP<Teko::InverseFactory> directSolveFactory = invLib->getInverseFactory("Amesos");

   {
      Teko::LinearOp probedInverse = Teko::buildInverse(*proberFactory,lo);
      Teko::LinearOp invLo = Teko::buildInverse(*directSolveFactory,lo);

      Thyra::LinearOpTester<double> tester;
      tester.dump_all(true);
      tester.show_all_tests(true);

      {
         const bool result = tester.compare( *probedInverse, *invLo, Teuchos::ptrFromRef(out));
         if (!result) {
            out << "Apply: FAILURE" << std::endl;
            success = false;
         }
         else
            out << "Apply: SUCCESS" << std::endl;
      }
   }
}

TEUCHOS_UNIT_TEST(tProbingFactory, callback_interface)
{
   // build global (or serial communicator)
   #ifdef HAVE_MPI
      Epetra_MpiComm Comm(MPI_COMM_WORLD);
   #else
      Epetra_SerialComm Comm;
   #endif

   // this should be tested!
}*/

#else

TEUCHOS_UNIT_TEST(tProbingFactory, no_isoroppia_available)
{
}

#endif
