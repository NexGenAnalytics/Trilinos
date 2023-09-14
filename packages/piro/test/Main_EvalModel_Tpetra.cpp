// @HEADER
// ************************************************************************
// 
//        Piro: Strategy package for embedded analysis capabilitites
//                  Copyright (2010) Sandia Corporation
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
// Questions? Contact Andy Salinger (agsalin@sandia.gov), Sandia
// National Laboratories.
// 
// ************************************************************************
// @HEADER

#include <iostream>
#include <string>

#include "MockModelEval_A_Tpetra.hpp"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <Thyra_ModelEvaluator.hpp>
#include <Thyra_ModelEvaluatorBase_decl.hpp>
#include <Thyra_TpetraVector_decl.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

template<class ScalarType>
int run(int argc, char *argv[])
{
  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using OP = typename Tpetra::Operator<ST,LO,GO,NT>;
  using MV = typename Tpetra::MultiVector<ST,LO,GO,NT>;
    
  using tcrsmatrix_t  = Tpetra::CrsMatrix<ST,LO,GO,NT>;

  using tvector_t = Thyra::TpetraVector<ST,LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;

  int status = 0;       // 0 = pass, failures are incremented
  bool success = true;  

  // Initialize MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int Proc = mpiSession.getRank();
  auto appComm = Tpetra::getDefaultComm();

  try
  {
    const RCP<Thyra::ModelEvaluator<ST>> Model = rcp(new MockModelEval_A_Tpetra(appComm));

    // Set input arguments to evalModel call
    Thyra::ModelEvaluatorBase::InArgs inArgs = Model->createInArgs();
    RCP<tvector_t> x = rcp(new tvector_t(/*Teuchos::null*/));
    inArgs.set_x(x);

    // Number of *vectors* of parameters
    int num_p = inArgs.Np();  
    RCP<tvector_t> p1;
    if (num_p > 0) {
      p1 = rcp(new tvector_t(/*Teuchos::null*/));
      inArgs.set_p(0, p1);
    }

    // Number of parameters in p1 vector
    int numParams = p1->getTpetraVector()->getLocalLength(); 

    // Set output arguments to evalModel call
    Thyra::ModelEvaluatorBase::OutArgs outArgs = Model->createOutArgs();
    // TO FIX & UNCOMMENT: RCP<tvector_t> f;
    // TO FIX & UNCOMMENT: f.initialize(/*XXXXX*/, x->getTpetraVector()->getMap());
    // TO FIX & UNCOMMENT: outArgs.set_f(f);  

    // Number of *vectors* of responses
    int num_g = outArgs.Ng(); 
    Thyra::ModelEvaluatorBase::Evaluation<Thyra::VectorBase<ST>> g1(Teuchos::null);
    if (num_g > 0) {
      outArgs.set_g(0, g1);
    }

    // Create a LinearOpWithSolveBase object for W to be evaluated
    // TO FIX & UNCOMMENT: RCP<OP> W_op = Model->create_W();
    // TO UNCOMMENT: outArgs.set_W(W_op);
    
    // TO FIX & UNCOMMENT: RCP<MV> dfdp = rcp(new MV(Teuchos::null, numParams));
    // TO UNCOMMENT: outArgs.set_DfDp(0, dfdp);

    // TO FIX & UNCOMMENT: RCP<MV> dgdp = rcp(new MV(g1->getMap(), numParams));
    // TO UNCOMMENT: outArgs.set_DgDp(0, 0, dgdp);
    
    // TO FIX & UNCOMMENT: RCP<MV> dgdx = rcp(new MV(x->getMap(), g1->getLocalLength()));
    // TO UNCOMMENT: outArgs.set_DgDx(0, dgdx);
    
    // Now, evaluate the model!
    Model->evalModel(inArgs, outArgs);
     
    // Print out everything
    if (Proc == 0)
      std::cout << "Finished Model Evaluation: Printing everything {Exact in brackets}" 
        << "\n-----------------------------------------------------------------"
        << std::setprecision(9) << std::endl;
      x->getTpetraVector()->print(std::cout << "\nSolution vector! {3,3,3,3}\n");
      if (num_p>0) p1->getTpetraVector()->print(std::cout << "\nParameters! {1,1}\n");
      // TO FIX & UNCOMMENT:f->print(std::cout << "\nResidual! {8,5,0,-7}\n");
      // TO FIX & UNCOMMENT:if (num_g>0) g1->print(std::cout << "\nResponses! {2}\n");
      // TO UNCOMMENT: RCP<tcrsmatrix_t> W = Teuchos::rcp_dynamic_cast<tcrsmatrix_t>(W_op, true);
      // TO UNCOMMENT: W->print(std::cout << "\nJacobian! {6 on diags}\n");
      // TO FIX & UNCOMMENT: dfdp.describe(std::cout << "\nDfDp sensitivity MultiVector! {-1,0,0,0}{0,-4,-6,-8}\n"); 
      // TO FIX & UNCOMMENT: dgdp.print(std::cout << "\nDgDp response sensitivity MultiVector!{2,2}\n");
      // TO FIX & UNCOMMENT: dgdx->print(std::cout << "\nDgDx^T response gradient MultiVector! {-2,-2,-2,-2}\n");

      if (Proc == 0)
        std::cout << "\n-----------------------------------------------------------------\n";
  
  } // try
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  
  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // run with different scalar types
  return run<double>(argc, argv);
  // return run<float>(argc, argv);
}
