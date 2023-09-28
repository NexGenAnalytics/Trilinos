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

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

// Thyra testing tools
#include "Thyra_TestingTools.hpp"
#include "Thyra_LinearOpTester.hpp"

// Thyra includes
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_ProductVectorBase.hpp"
#include "Thyra_SpmdVectorSpaceBase.hpp"
#include "Thyra_DetachedSpmdVectorView.hpp"

// TriUtils includes
//#include "Trilinos_Util_CrsMatrixGallery.h" // TO REMOVE EPETRA

// Teko includes
#include "Teko_StridedTpetraOperator.hpp"
#include "Teko_Utilities.hpp"

#define SS_ECHO(ops) { std::stringstream ss; ss << ops; ECHO(ss.str()); };

// Teuchos using
using Teuchos::Comm;
using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;

// Thyra using
using Thyra::VectorBase;
using Thyra::LinearOpBase;
using Thyra::createMember;
using Thyra::LinearOpTester;

// Tpetra using
using ST = typename Tpetra::Vector<double>::scalar_type;
using LO = typename Tpetra::Vector<>::local_ordinal_type;
using GO = typename Tpetra::Vector<>::global_ordinal_type;
using NT = typename Tpetra::Vector<>::node_type;
using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;
using tmap_t = Tpetra::Map<LO,GO,NT>;
using tmultivector_t = Tpetra::MultiVector<ST,LO,GO,NT>;

ST tolerance = 1e-14;

const RCP<const Comm<int>> GetComm()
{
  return Tpetra::getDefaultComm();
}

TEUCHOS_UNIT_TEST(tStridedTpetraOperator, test_numvars_constr)
{
  // communicator
  auto comm = GetComm();
  SS_ECHO("tStridedTpetraOperator::test_numvars: " << "Running on " << comm->getSize() << " processors");

  // pick 
  int nx = 3 * 25 * comm->getSize();
  int ny = 3 * 50 * comm->getSize();

  // create a big matrix to play with
  RCP<tmap_t> FGallery = rcp(new tmap_t(nx * ny, 0, comm));
  RCP<tcrsmatrix_t> A = rcp(new tcrsmatrix_t(FGallery, 1));
  ST beforeNorm = A->getFrobeniusNorm();    

  // create a strided tpetra operator
  int vars = 3;
  int width = 3;
  tmultivector_t x(A->getDomainMap(), width);
  tmultivector_t ys(A->getRangeMap(), width);
  tmultivector_t y(A->getRangeMap(), width);
  Teko::TpetraHelpers::StridedTpetraOperator shell(vars, A);  // AM: TO FIX

  // test the operator against a lot of random vectors
  int numtests = 50;
  ST max = 0.0;
  ST min = 1.0;
  for(int i=0; i<numtests; i++) {
    std::vector<ST> norm(width);    
    std::vector<ST> rel(width); 
    x.randomize();                                            // AM: TO FIX    
  
    shell.apply(x,y);                                         // AM: TO FIX
    A->apply(x,ys);

    tmultivector_t e(y, Teuchos::Copy);
    e.update(-1.0, ys, 1.0);
    e.norm2(Teuchos::ArrayView<ST>(norm));

    // compute relative error
    ys.norm2(Teuchos::ArrayView<ST>(rel));
    for(int j=0;j<width;j++) {
      max = max>norm[j]/rel[j] ? max : norm[j]/rel[j];
      min = min<norm[j]/rel[j] ? min : norm[j]/rel[j];
    }
  }
  TEST_ASSERT(max >= min);
  TEST_ASSERT(max <= tolerance);
  
  // double everything
  A->scale(2.0);

  ST afterNorm = A->getFrobeniusNorm();
  TEST_ASSERT(beforeNorm != afterNorm);

  shell.RebuildOps();

  // test the operator against a lot of random vectors
  numtests = 50;
  max = 0.0;
  min = 1.0;
  for(int i=0; i<numtests; i++) {
    std::vector<ST> norm(width);
    std::vector<ST> rel(width);
    x.randomize();                                            // AM: PROBABLY TO FIX          

    shell.apply(x,y);                                         // AM: PROBABLY TO FIX
    A->apply(x,ys);

    tmultivector_t e(y, Teuchos::Copy);
    e.update(-1.0, ys, 1.0);
    e.norm2(Teuchos::ArrayView<ST>(norm));

    // compute relative error
    ys.norm2(Teuchos::ArrayView<ST>(rel));
    for(int j=0; j<width; j++) {
      max = max>norm[j]/rel[j] ? max : norm[j]/rel[j];
      min = min<norm[j]/rel[j] ? min : norm[j]/rel[j];
    }
  }
  TEST_ASSERT(max >= min);
  TEST_ASSERT(max <= tolerance);
}

/*
TEUCHOS_UNIT_TEST(tStridedTpetraOperator, test_vector_constr)
{
   const Epetra_Comm & comm = *GetComm();

   SS_ECHO("\n   tStridedTpetraOperator::test_vector_constr: "
         << "Running on " << comm.NumProc() << " processors");

   // pick 
   int nx = 3 * 25 * comm.NumProc();
   int ny = 3 * 50 * comm.NumProc();


   // create a big matrix to play with
   // note: this matrix is not really strided
   //       however, I just need a nontrivial
   //       matrix to play with
   Trilinos_Util::CrsMatrixGallery FGallery("recirc_2d",comm,false); // CJ TODO FIXME: change for Epetra64
   FGallery.Set("nx",nx);
   FGallery.Set("ny",ny);
   RCP<Epetra_CrsMatrix> A = rcp(FGallery.GetMatrix(),false);

   double beforeNorm = A->NormOne();

   int width = 3;
   Epetra_MultiVector x(A->OperatorDomainMap(),width);
   Epetra_MultiVector ys(A->OperatorRangeMap(),width);
   Epetra_MultiVector y(A->OperatorRangeMap(),width);

   std::vector<int> vars;
   vars.push_back(2);
   vars.push_back(1);
   Teko::Epetra::StridedEpetraOperator shell(vars,A);

   // test the operator against a lot of random vectors
   int numtests = 50;
   double max = 0.0;
   double min = 1.0;
   for(int i=0;i<numtests;i++) {
      std::vector<double> norm(width);
      std::vector<double> rel(width);
      x.Random();

      shell.Apply(x,y);
      A->Apply(x,ys);

      Epetra_MultiVector e(y);
      e.Update(-1.0,ys,1.0);
      e.Norm2(&norm[0]);

      // compute relative error
      ys.Norm2(&rel[0]);
      for(int j=0;j<width;j++) {
         max = max>norm[j]/rel[j] ? max : norm[j]/rel[j];
         min = min<norm[j]/rel[j] ? min : norm[j]/rel[j];
      }
   }
   TEST_ASSERT(max>=min);
   TEST_ASSERT(max<=tolerance)

   int * indexOffset,* indicies;
   double * values;
   A->ExtractCrsDataPointers(indexOffset,indicies,values);
   for(int i=0;i<A->NumMyNonzeros();i++)
      values[i] *= 2.0; // square everything!

   double afterNorm = A->NormOne();
   TEST_ASSERT(beforeNorm!=afterNorm);

   shell.RebuildOps();

   // test the operator against a lot of random vectors
   numtests = 50;
   max = 0.0;
   min = 1.0;
   for(int i=0;i<numtests;i++) {
      std::vector<double> norm(width);
      std::vector<double> rel(width);
      x.Random();

      shell.Apply(x,y);
      A->Apply(x,ys);

      Epetra_MultiVector e(y);
      e.Update(-1.0,ys,1.0);
      e.Norm2(&norm[0]);

      // compute relative error
      ys.Norm2(&rel[0]);
      for(int j=0;j<width;j++) {
         max = max>norm[j]/rel[j] ? max : norm[j]/rel[j];
         min = min<norm[j]/rel[j] ? min : norm[j]/rel[j];
      }
   }
   TEST_ASSERT(max>=min);
   TEST_ASSERT(max<=tolerance);
}

TEUCHOS_UNIT_TEST(tStridedTpetraOperator, test_reorder)
{
   const Epetra_Comm & comm = *GetComm();

   for(int total=0;total<3;total++) {
   
      std::string tstr = total ? "(composite reorder)" : "(flat reorder)";
   
      SS_ECHO("\n   tStridedTpetraOperator::test_reorder" << tstr << ": "
            << "Running on " << comm.NumProc() << " processors");
   
      // pick 
      int nx = 3 * 25 * comm.NumProc();
      int ny = 3 * 50 * comm.NumProc();
   
      // create a big matrix to play with
      // note: this matrix is not really strided
      //       however, I just need a nontrivial
      //       matrix to play with
      Trilinos_Util::CrsMatrixGallery FGallery("recirc_2d",comm,false); // CJ TODO FIXME: change for Epetra64
      FGallery.Set("nx",nx);
      FGallery.Set("ny",ny);
      RCP<Epetra_CrsMatrix> A = rcp(FGallery.GetMatrix(),false);
   
      int width = 3;
      Epetra_MultiVector x(A->OperatorDomainMap(),width);
      Epetra_MultiVector yf(A->OperatorRangeMap(),width);
      Epetra_MultiVector yr(A->OperatorRangeMap(),width);
   
      Teko::Epetra::StridedEpetraOperator flatShell(3,A,"Af");
      Teko::Epetra::StridedEpetraOperator reorderShell(3,A,"Ar");
    
      Teko::BlockReorderManager brm;
      switch (total) {
      case 0:
         brm.SetNumBlocks(3);
         brm.SetBlock(0,1);
         brm.SetBlock(1,0);
         brm.SetBlock(2,2);
         break;
      case 1:
         brm.SetNumBlocks(2);
         brm.SetBlock(0,1);
         brm.GetBlock(1)->SetNumBlocks(2);
         brm.GetBlock(1)->SetBlock(0,0);
         brm.GetBlock(1)->SetBlock(1,2);
         break;
      case 2:
         brm.SetNumBlocks(2);
         brm.GetBlock(0)->SetNumBlocks(2);
         brm.GetBlock(0)->SetBlock(0,0);
         brm.GetBlock(0)->SetBlock(1,2);
         brm.SetBlock(1,1);
         break;
      }
      reorderShell.Reorder(brm);
      SS_ECHO("\n   tStridedTpetraOperator::test_reorder" << tstr << ": patern = " << brm.toString());
   
      SS_ECHO("\n   tStridedTpetraOperator::test_reorder" << tstr << ":\n");
      SS_ECHO("\n      " << Teuchos::describe(*reorderShell.getThyraOp(), Teuchos::VERB_HIGH)  << std::endl);
   
      // test the operator against a lot of random vectors
      int numtests = 10;
      double max = 0.0;
      double min = 1.0;
      for(int i=0;i<numtests;i++) {
         std::vector<double> norm(width);
         std::vector<double> rel(width);
         x.Random();
   
         flatShell.Apply(x,yf);
         reorderShell.Apply(x,yr);
   
         Epetra_MultiVector e(yf);
         e.Update(-1.0,yr,1.0);
         e.Norm2(&norm[0]);
   
         // compute relative error
         yf.Norm2(&rel[0]);
         for(int j=0;j<width;j++) {
            max = max>norm[j]/rel[j] ? max : norm[j]/rel[j];
            min = min<norm[j]/rel[j] ? min : norm[j]/rel[j];
         }
      }
      TEST_ASSERT(max>=min);
      TEST_ASSERT(max<=tolerance);
   }
}

TEUCHOS_UNIT_TEST(tStridedTpetraOperator, test_print_norm)
{
   const Epetra_Comm & comm = *GetComm();

   // pick 
   int nx = 3 * 25 * comm.NumProc();
   int ny = 3 * 50 * comm.NumProc();


   // create a big matrix to play with
   // note: this matrix is not really strided
   //       however, I just need a nontrivial
   //       matrix to play with
   Trilinos_Util::CrsMatrixGallery FGallery("recirc_2d",comm,false); // CJ TODO FIXME: change for Epetra64
   FGallery.Set("nx",nx);
   FGallery.Set("ny",ny);
   RCP<Epetra_CrsMatrix> A = rcp(FGallery.GetMatrix(),false);

   std::vector<int> vars;
   vars.push_back(1);
   vars.push_back(1);
   vars.push_back(1);
   Teko::Epetra::StridedEpetraOperator shell(vars,A);

   std::string normString = shell.PrintNorm();
   *Teko::getOutputStream() << std::endl << normString << std::endl;
}
*/