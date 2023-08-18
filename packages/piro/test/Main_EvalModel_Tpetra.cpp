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
  using Scalar = double;

  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using OP = typename Tpetra::Operator<Scalar,LO,GO,NT>;
  using MV = typename Tpetra::MultiVector<Scalar,LO,GO,NT>;

  using Tpetra_Vector = Tpetra::Vector<Scalar>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<Scalar,LO,GO,NT>;
  using MEB = Thyra::ModelEvaluatorBase;


  using Teuchos::RCP;
  using Teuchos::rcp;

  int status = 0;       // 0 = pass, failures are incremented
  bool success = true;  

  // Initialize MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int Proc = mpiSession.getRank();
  auto appComm = Tpetra::getDefaultComm();

  try {
    // TODO: Display like for Epetra version (currently minimal working)
    const RCP<Thyra::ModelEvaluator<Scalar>> me = rcp(new MockModelEval_A_Tpetra(appComm));

    MEB::Evaluation<Thyra::VectorBase<double> > g = Thyra::createMember(*me->get_g_space(0));
    MEB::Evaluation<Thyra::VectorBase<double> > x = Thyra::createMember(*me->get_x_space());

    MEB::InArgs<Scalar>  in_args = me->createInArgs();
    in_args.set_x(x);

    MEB::OutArgs<Scalar> out_args = me->createOutArgs();
    out_args.set_g(0,g);

    me->evalModel(in_args, out_args);

    // Print out everything
    if (Proc == 0)
      std::cout << "Finished Model Evaluation: Printing everything {Exact in brackets}" 
        << "\n-----------------------------------------------------------------"
        << std::setprecision(9) << std::endl;

      // TODO: Display like for Epetra version

      if (Proc == 0)
      std::cout << "\n-----------------------------------------------------------------\n";
  } // try
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  
  if (Proc==0) {
    if (status==0) 
      std::cout << "TEST PASSED" << std::endl;
    else 
      std::cout << "TEST Failed" << std::endl;
  }

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // to change Scalar we should also template the Model class
  // run with different scalar types
  // run<double>(argc, argv);
  // run<float>(argc, argv);
  
  // run with double scalar type 
  run(argc, argv);
}