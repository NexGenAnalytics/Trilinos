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

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Thyra_ModelEvaluator.hpp"

#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

int run(int argc, char *argv[]) {
  // Model is currently supporting double
  using ScalarType = double;

  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using OP = typename Tpetra::Operator<ScalarType,LO,GO,NT>;
  using MV = typename Tpetra::MultiVector<ScalarType,LO,GO,NT>;

  using Tpetra_Vector = Tpetra::Vector<ScalarType>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<ScalarType,LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;

  int status = 0;       // 0 = pass, failures are incremented
  bool success = true;  

  // Initialize MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int Proc = mpiSession.getRank();
  auto appComm = Tpetra::getDefaultComm();

  try {

    const RCP<Thyra::ModelEvaluator<ScalarType>> Model = rcp(new MockModelEval_A_Tpetra(appComm));

    // Set input arguments to evalModel call
    Thyra::ModelEvaluatorBase::InArgs<ScalarType> inArgs = Model->createInArgs();
    RCP<Tpetra_Vector> x = rcp(new Tpetra_Vector(Teuchos::null));
    inArgs.set_x(x);
    int num_p = inArgs.Np(); // Number of *vectors* of parameters 
    RCP<Tpetra_Vector> p1;
    if (num_p > 0) {
      p1 = rcp(new Tpetra_Vector(Teuchos::null));
      inArgs.set_p(0, p1);
    }
    int numParams = p1.getLocalLength(); // Number of parameters in p1 vector

    // Set output arguments to evalModel call
    Thyra::ModelEvaluatorBase::OutArgs<ScalarType> outArgs = Model->createOutArgs();
    RCP<Tpetra_Vector> f = rcp(new Tpetra_Vector(x.getMap()));
    outArgs.set_f(f);
    int num_g = outArgs.Ng(); // Number of *vectors* of responses
    RCP<Tpetra_Vector> g1;
    if (num_g > 0) {
      g1 = rcp(new Tpetra_Vector(Teuchos::null));
      outArgs.set_g(0, g1);
    }

    // Create a LinearOpWithSolveBase object for W to be evaluated
    RCP<OP> W_op = Model->create_W();
    outArgs.set_W(W_op);

    RCP<MV> dfdp = rcp(new MV(Teuchos::null, numParams));
    outArgs.set_DfDp(0, dfdp);
    RCP<MV> dgdp = rcp(new MV(g1->getMap(), numParams));
    outArgs.set_DgDp(0, 0, dgdp);
    RCP<MV> dgdx = rcp(new MV(x->getMap(), g1->getLocalLength()));
    outArgs.set_DgDx(0, dgdx);

    // Now, evaluate the model!
    Model->evalModel(inArgs, outArgs);

    // Print out everything
    if (Proc == 0)
      std::cout << "Finished Model Evaluation: Printing everything {Exact in brackets}" 
        << "\n-----------------------------------------------------------------"
        << std::setprecision(9) << std::endl;
      x.print(std::cout << "\nSolution vector! {3,3,3,3}\n");
      if (num_p>0) p1->print(std::cout << "\nParameters! {1,1}\n");
      f.print(std::cout << "\nResidual! {8,5,0,-7}\n");
      if (num_g>0) g1->print(std::cout << "\nResponses! {2}\n");
      RCP<Tpetra_Matrix> W = Teuchos::rcp_dynamic_cast<Tpetra_Matrix>(W_op, true);
      W.print(std::cout << "\nJacobian! {6 on diags}\n");
      dfdp.print(std::cout << "\nDfDp sensitivity MultiVector! {-1,0,0,0}{0,-4,-6,-8}\n");
      dgdp.print(std::cout << "\nDgDp response sensitivity MultiVector!{2,2}\n");
      dgdx.print(std::cout << "\nDgDx^T response gradient MultiVector! {-2,-2,-2,-2}\n");

      if (Proc == 0)
      std::cout << "\n-----------------------------------------------------------------\n";
  } // try
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  
  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // to change ScalarType we should also template the Model class
  // run with different scalar types
  // run<double>(argc, argv);
  // run<float>(argc, argv);
  
  // run with double scalar type 
  run();
}