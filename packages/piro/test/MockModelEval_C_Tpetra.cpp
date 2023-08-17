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

#include "MockModelEval_C_Tpetra.hpp"
#include "Piro_ConfigDefs.hpp"

// #include "Epetra_LocalMap.h" // is in epetra test
// #include "Epetra_CrsMatrix.h" // is in epetra test

// #ifdef HAVE_PIRO_STOKHOS  // is in epetra test
// #include "Stokhos_Epetra.hpp"  // is in epetra test
// #endif // is in epetra test

using Teuchos::RCP;
using Teuchos::rcp;

MockModelEval_C_Tpetra::MockModelEval_C_Tpetra(const Teuchos::RCP<const Teuchos::Comm<int> >  comm_) :
  comm(comm_)
{
  //set up map and initial guess for solution vector
  const int vecLength = 1;
  x_map = rcp(new Tpetra_Map(vecLength, 0, comm));
  x_init = rcp(new Tpetra_Vector(x_map));
  x_init->putScalar(1.0);
  x_dot_init = rcp(new Tpetra_Vector(x_map));
  x_dot_init->putScalar(0.0);

  // WIP: previous line done from epetra to tpetra. Continue with next lines

  //set up responses
  const int numResponses = 1;
  g_map = rcp(new Epetra_LocalMap(numResponses, 0, *comm));

  //set up parameters
  const int numParameters = 1;
  p_map = rcp(new const Tpetra_Map(numParameters, 0, comm, Tpetra::LocallyReplicated));
  p_init = rcp(new Tpetra_Vector(p_map));
  for (int i=0; i<numParameters; i++) (*p_init)[i]= i+1;
  
  //setup Jacobian graph
  graph = rcp(new Epetra_CrsGraph(Copy, *x_map, 1));
  for (int i=0; i<vecLength; i++)
    graph->InsertGlobalIndices(i,1,&i);
  graph->FillComplete();
}

MockModelEval_C_Tpetra::~MockModelEval_C()
{
}

Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
MockModelEval_C_Tpetra::get_x_space() const
{
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> x_space =
      Thyra::createVectorSpace<double>(x_map);
  return x_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
MockModelEval_C_Tpetra::get_f_space() const
{
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> f_space =
      Thyra::createVectorSpace<double>(x_map);
  return f_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
MockModelEval_C_Tpetra::get_p_space(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l != 0, std::logic_error,
                     std::endl <<
                     "Error!  MockModelEval_C_Tpetra::get_p_space() only " <<
                     " supports 1 parameter vector.  Supplied index l = " <<
                     l << std::endl);
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> p_space =
        Thyra::createVectorSpace<double>(p_map);
  return p_space;
}

Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
MockModelEval_C_Tpetra::get_g_space(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l != 0, std::logic_error,
                     std::endl <<
                     "Error!  MockModelEval_C_Tpetra::get_g_space() only " <<
                     " supports 1 response.  Supplied index l = " <<
                     l << std::endl);
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> g_space =
        Thyra::createVectorSpace<double>(g_map);
  return g_space;
}

RCP<const  Teuchos::Array<std::string> > MockModelEval_C_Tpetra::get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l != 0, std::logic_error,
                     std::endl <<
                     "Error!  MockModelEval_C_Tpetra::get_p_names() only " <<
                     " supports 1 parameter vector.  Supplied index l = " <<
                     l << std::endl);

  Teuchos::Ordinal num_p = p_map->getLocalNumElements();
  RCP<Teuchos::Array<std::string> > p_names =
      rcp(new Teuchos::Array<std::string>(num_p) );
  for (int i=0; i<num_p; i++) {
    std::stringstream ss;
    ss << "Parameter " << i;
    const std::string name = ss.str();
    (*p_names)[i] = name;
  }
  return p_names;
}

Teuchos::RCP<Thyra::LinearOpBase<double>>
MockModelEval_C_Tpetra::create_W_op() const
{
  const Teuchos::RCP<Tpetra_Operator> W =
      Teuchos::rcp(new Tpetra_CrsMatrix(graph));
  return Thyra::createLinearOp(W);
}

//! Create preconditioner operator
Teuchos::RCP<Thyra::PreconditionerBase<double>>
MockModelEval_C_Tpetra::create_W_prec() const
{
  return Teuchos::null;
}

// TODO: following is epetra version of the impementation remaining to convert for tpetra
/*
EpetraExt::ModelEvaluator::InArgs 
MockModelEval_C::createInArgs() const
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.set_Np(1);
  inArgs.setSupports(IN_ARG_x, true);

#ifdef HAVE_PIRO_STOKHOS
  inArgs.setSupports(IN_ARG_x_sg, true);
  inArgs.setSupports(IN_ARG_x_dot_sg, true);
  inArgs.setSupports(IN_ARG_p_sg, 0, true); // 1 SG parameter vector
  inArgs.setSupports(IN_ARG_sg_basis, true);
  inArgs.setSupports(IN_ARG_sg_quadrature, true);
  inArgs.setSupports(IN_ARG_sg_expansion, true);
#endif

  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs 
MockModelEval_C::createOutArgs() const
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.set_Np_Ng(1, 1);

  outArgs.setSupports(OUT_ARG_f, true);
  outArgs.setSupports(OUT_ARG_W, true);
  outArgs.set_W_properties( DerivativeProperties(
      DERIV_LINEARITY_UNKNOWN, DERIV_RANK_FULL, true));
  outArgs.setSupports(OUT_ARG_DfDp, 0, DERIV_MV_BY_COL);
  outArgs.setSupports(OUT_ARG_DgDx, 0, DERIV_TRANS_MV_BY_ROW);
  outArgs.setSupports(OUT_ARG_DgDp, 0, 0, DERIV_MV_BY_COL);

#ifdef HAVE_PIRO_STOKHOS
  outArgs.setSupports(OUT_ARG_f_sg, true);
  outArgs.setSupports(OUT_ARG_W_sg, true);
  outArgs.setSupports(OUT_ARG_g_sg, 0, true);
  outArgs.setSupports(OUT_ARG_DfDp_sg, 0, DERIV_MV_BY_COL);
  outArgs.setSupports(OUT_ARG_DgDx_sg, 0, DERIV_TRANS_MV_BY_ROW);
  outArgs.setSupports(OUT_ARG_DgDp_sg, 0, 0, DERIV_MV_BY_COL);
#endif

  return outArgs;
}

void 
MockModelEval_C::evalModel(const InArgs& inArgs, const OutArgs& outArgs) const
{
  int proc = comm->MyPID();

  // 
  // Deterministic calculation
  //

  // Parse InArgs
  RCP<const Epetra_Vector> p_in = inArgs.get_p(0);
  if (p_in == Teuchos::null)
    p_in = p_init;

  RCP<const Epetra_Vector> x_in = inArgs.get_x();

  // Parse OutArgs
  RCP<Epetra_Vector> f_out = outArgs.get_f(); 
  if (f_out != Teuchos::null) {
    double p = (*p_in)[0];
    if (proc == 0) {
      double x = (*x_in)[0];
      (*f_out)[0] = 0.5*(x*x - p*p);
    }
  }

  RCP<Epetra_CrsMatrix> W_out = 
    Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(outArgs.get_W()); 
  if (W_out != Teuchos::null) {
    if (proc == 0) {
      double x = (*x_in)[0];
      int i = 0;
      W_out->ReplaceMyValues(i, 1, &x, &i);
    }
  }

  RCP<Epetra_MultiVector> dfdp = outArgs.get_DfDp(0).getMultiVector();
  if (dfdp != Teuchos::null) {
    double p = (*p_in)[0];
    if (proc == 0)
      (*dfdp)[0][0] = -p;
  }

  RCP<Epetra_Vector> g_out = outArgs.get_g(0); 
  if (g_out != Teuchos::null) {
    double p = (*p_in)[0];
    if (proc == 0) {
      double x = (*x_in)[0];
      (*g_out)[0] = 0.5*(x*x + p*p);
    }
  }
    

  RCP<Epetra_MultiVector> dgdx = outArgs.get_DgDx(0).getMultiVector();
  if (dgdx != Teuchos::null) {
    if (proc == 0) {
      double x = (*x_in)[0];
      (*dgdx)[0][0] = x;
    }
  }

  RCP<Epetra_MultiVector> dgdp = outArgs.get_DgDp(0,0).getMultiVector();
  if (dgdp != Teuchos::null) {
    double p = (*p_in)[0];
    if (proc == 0) {
      (*dgdp)[0][0] = p;
    }
  }

  // 
  // Stochastic calculation
  //

#ifdef HAVE_PIRO_STOKHOS
  // Parse InArgs
  RCP<const Stokhos::OrthogPolyBasis<int,double> > basis = 
    inArgs.get_sg_basis();
  RCP<Stokhos::OrthogPolyExpansion<int,double> > expn = 
    inArgs.get_sg_expansion();
  InArgs::sg_const_vector_t x_sg = inArgs.get_x_sg();
  InArgs::sg_const_vector_t p_sg = inArgs.get_p_sg(0);

  Stokhos::OrthogPolyApprox<int,double> x(basis), x2(basis);
  if (x_sg != Teuchos::null && proc == 0) {
    for (int i=0; i<basis->size(); i++) {
      x[i] = (*x_sg)[i][0];
    }
    expn->times(x2, x, x);
  }

  Stokhos::OrthogPolyApprox<int,double> p(basis), p2(basis);
  if (p_sg != Teuchos::null) {
    for (int i=0; i<basis->size(); i++) {
      p[i] = (*p_sg)[i][0];
    }
    expn->times(p2, p, p);
  }

  // Parse OutArgs
  OutArgs::sg_vector_t f_sg = outArgs.get_f_sg();
  if (f_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<f_sg->size(); block++) {
      (*f_sg)[block][0] = 0.5*(x2[block] - p2[block]);
    }
  }

  OutArgs::sg_operator_t W_sg = outArgs.get_W_sg();
  if (W_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<W_sg->size(); block++) {
      Teuchos::RCP<Epetra_CrsMatrix> W = 
	Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_sg->getCoeffPtr(block), 
						    true);
      int i = 0;
      int ret = W->ReplaceMyValues(i, 1, &x[block], &i);
      if (ret != 0)
	std::cout << "ReplaceMyValues returned " << ret << "!" << std::endl;
    }
  }

  RCP<Stokhos::EpetraMultiVectorOrthogPoly> dfdp_sg = 
    outArgs.get_DfDp_sg(0).getMultiVector();
  if (dfdp_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<dfdp_sg->size(); block++) {
      (*dfdp_sg)[block][0][0] = -p[block];
    }
  }

  OutArgs::sg_vector_t g_sg = outArgs.get_g_sg(0); 
  if (g_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<g_sg->size(); block++) {
      (*g_sg)[block][0] = 0.5*(x2[block] + p2[block]);
    }
  }

  RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdx_sg = 
    outArgs.get_DgDx_sg(0).getMultiVector();
  if (dgdx_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<dgdx_sg->size(); block++) {
      (*dgdx_sg)[block][0][0] = x[block];
    }
  }

  RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdp_sg = 
    outArgs.get_DgDp_sg(0,0).getMultiVector();
  if (dgdp_sg != Teuchos::null && proc == 0) {
    for (int block=0; block<dgdp_sg->size(); block++) {
      (*dgdp_sg)[block][0][0] = p[block];
    }
  }
#endif*/
} 
