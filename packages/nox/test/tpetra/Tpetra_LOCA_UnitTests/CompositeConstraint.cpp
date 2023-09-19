// $Id$
// $Source$

//@HEADER
// ************************************************************************
//
//            LOCA: Library of Continuation Algorithms Package
//                 Copyright (2005) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov) or
// Eric Phipps (etphipp@sandia.gov), Sandia National Laboratories.
// ************************************************************************
//  CVS Information
//  $Source$
//  $Author$
//  $Date$
//  $Revision$
// ************************************************************************
//@HEADER

#include "LOCA.H"
// #include "LOCA_Epetra.H" // CWS: find replacement

#include "LOCA_MultiContinuation_CompositeConstraint.H"
#include "NOX_TestCompare.H"

#include "NOX_Thyra.H"

// Trilinos Objects
#include <Tpetra_Map.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <Tpetra_CrsMatrix.hpp>

// User's application specific files
#include "LOCALinearConstraint.H"

template <typename ScalarType>
int run(int argc, char *argv[]) {

  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NO = typename Tpetra::Map<>::node_type;

  using tmap_t       = typename Tpetra::Map<ST,LO,GO>;
  using tvector_t    = typename Tpetra::Vector<ST,LO,GO,NT>
  using tcrsmatrix_t = typename Tpetra::CrsMatrix<ST,LO,GO>;
  using tcrsgraph_t  = typename Tpetra::CrsGraph<LO,GO,NO>;

  int n = 10;
  int ierr = 0;
  ST reltol = 1.0e-14;
  ST abstol = 1.0e-14;
  int MyPID = 0;

  try {

    // Create a communicator for Tpetra objects
    Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
    const auto Comm = Tpetra::getDefaultComm();

    MyPID = Comm->getRank();

    // Create the map
    tmap_t map(n, 0, Comm);

    bool verbose = false;
    // Check for verbose output
    if (argc>1)
      if (argv[1][0]=='-' && argv[1][1]=='v')
    verbose = true;

    // Seed the random number generator in Teuchos.  We create random
    // bordering matrices and it is possible different processors might generate
    // different matrices.  By setting the seed, this shouldn't happen.
    Teuchos::ScalarTraits<ST>::seedrandom(12345);

    // Create and initialize the parameter vector
    LOCA::ParameterVector pVector;
    pVector.addParameter("Param 1",  1.69);
    pVector.addParameter("Param 2", -9.7);
    pVector.addParameter("Param 3",  0.35);
    pVector.addParameter("Param 4", -0.78);
    pVector.addParameter("Param 5",  2.53);

    // Create parameter list
    Teuchos::RCP<Teuchos::ParameterList> paramList =
      Teuchos::rcp(new Teuchos::ParameterList);

    Teuchos::ParameterList& nlParams = paramList->sublist("NOX");
    Teuchos::ParameterList& nlPrintParams = nlParams.sublist("Printing");
    nlPrintParams.set("MyPID", MyPID);
    if (verbose)
       nlPrintParams.set("Output Information",
                  NOX::Utils::Error +
                  NOX::Utils::Details +
                  NOX::Utils::OuterIteration +
                  NOX::Utils::InnerIteration +
                  NOX::Utils::Warning +
                  NOX::Utils::TestDetails +
                  NOX::Utils::StepperIteration +
                  NOX::Utils::StepperDetails);
     else
       nlPrintParams.set("Output Information", NOX::Utils::Error);

    // Create global data object
    Teuchos::RCP<LOCA::GlobalData> globalData =
      LOCA::createGlobalData(paramList);

    // Teuchos::RCP<tvector_t> clone_vec = Teuchos::rcp (new tvector_t(map));
    tvector_t clone_vec(map);
    NOX::Thyra::Vector nox_clone_vec(clone_vec);

    Teuchos::RCP<NOX::Abstract::Vector> x =
      nox_clone_vec.clone(NOX::ShapeCopy);
    x->random();

    Teuchos::RCP<NOX::Abstract::MultiVector> dx1 =
      nox_clone_vec.createMultiVector(3);
    Teuchos::RCP<NOX::Abstract::MultiVector> dx2 =
      nox_clone_vec.createMultiVector(1);
    Teuchos::RCP<NOX::Abstract::MultiVector> dx3 =
      nox_clone_vec.createMultiVector(2);
    Teuchos::RCP<NOX::Abstract::MultiVector> dx4 =
      nox_clone_vec.createMultiVector(2);
    dx1->random();
    dx2->random();
    dx3->init(0.0);
    dx4->random();

    Teuchos::RCP<NOX::Abstract::MultiVector> dx_all =
      dx1->clone(NOX::DeepCopy);
    dx_all->augment(*dx2);
    dx_all->augment(*dx3);
    dx_all->augment(*dx4);

    NOX::Abstract::MultiVector::DenseMatrix dp1(dx1->numVectors(),
                        pVector.length());
    NOX::Abstract::MultiVector::DenseMatrix dp2(dx2->numVectors(),
                        pVector.length());
    NOX::Abstract::MultiVector::DenseMatrix dp3(dx3->numVectors(),
                        pVector.length());
    NOX::Abstract::MultiVector::DenseMatrix dp4(dx4->numVectors(),
                        pVector.length());
    dp1.random();
    dp2.random();
    dp3.random();
    dp4.random();

    NOX::Abstract::MultiVector::DenseMatrix dp_all(dx_all->numVectors(),
                           pVector.length());
    for (int j=0; j<dp_all.numCols(); j++) {
      for (int i=0; i<dp1.numRows(); i++)
    dp_all(i,j) = dp1(i,j);
      for (int i=0; i<dp2.numRows(); i++)
    dp_all(dp1.numRows()+i,j) = dp2(i,j);
      for (int i=0; i<dp3.numRows(); i++)
    dp_all(dp1.numRows()+dp2.numRows()+i,j) = dp3(i,j);
      for (int i=0; i<dp4.numRows(); i++)
    dp_all(dp1.numRows()+dp2.numRows()+dp3.numRows()+i,j) = dp4(i,j);
    }


    std::vector< Teuchos::RCP<LOCA::MultiContinuation::ConstraintInterface> > constraintObjs(4);
    Teuchos::RCP<LinearConstraint> linear_constraint;

    linear_constraint = Teuchos::rcp(new LinearConstraint(dx1->numVectors(),
                              pVector,
                              nox_clone_vec));
    linear_constraint->setDgDx(*dx1);
    linear_constraint->setDgDp(dp1);
    linear_constraint->setIsZeroDX(false);
    constraintObjs[0] = linear_constraint;

    linear_constraint = Teuchos::rcp(new LinearConstraint(dx2->numVectors(),
                              pVector,
                              nox_clone_vec));
    linear_constraint->setDgDx(*dx2);
    linear_constraint->setDgDp(dp2);
    linear_constraint->setIsZeroDX(false);
    constraintObjs[1] = linear_constraint;

    linear_constraint = Teuchos::rcp(new LinearConstraint(dx3->numVectors(),
                              pVector,
                              nox_clone_vec));
    linear_constraint->setDgDx(*dx3);
    linear_constraint->setDgDp(dp3);
    linear_constraint->setIsZeroDX(true);
    constraintObjs[2] = linear_constraint;

    linear_constraint = Teuchos::rcp(new LinearConstraint(dx4->numVectors(),
                              pVector,
                              nox_clone_vec));
    linear_constraint->setDgDx(*dx4);
    linear_constraint->setDgDp(dp4);
    linear_constraint->setIsZeroDX(false);
    constraintObjs[3] = linear_constraint;

    // Check some statistics on the solution
    NOX::TestCompare testCompare(globalData->locaUtils->out(),
                 *(globalData->locaUtils));

    LOCA::MultiContinuation::CompositeConstraint composite(globalData,
                               constraintObjs);
    composite.setX(*x);

    LinearConstraint combined(dx_all->numVectors(), pVector, nox_clone_vec);
    combined.setDgDx(*dx_all);
    combined.setDgDp(dp_all);
    combined.setX(*x);

    //
    // test computeConstraints()
    //

    composite.computeConstraints();
    combined.computeConstraints();

    int numConstraints = dx_all->numVectors();
    const NOX::Abstract::MultiVector::DenseMatrix& g_composite =
      composite.getConstraints();
    const NOX::Abstract::MultiVector::DenseMatrix& g_combined =
      combined.getConstraints();

    ierr += testCompare.testMatrix(
                 g_composite, g_combined, reltol, abstol,
                 "CompositeConstraint::computeConstraints()");

    //
    // test computeDP()
    //

    std::vector<int> paramIDs(3);
    paramIDs[0] = 1;
    paramIDs[1] = 2;
    paramIDs[2] = 4;
    NOX::Abstract::MultiVector::DenseMatrix dgdp_composite(
                            numConstraints,
                            paramIDs.size()+1);
    NOX::Abstract::MultiVector::DenseMatrix dgdp_combined(
                            numConstraints,
                            paramIDs.size()+1);
    dgdp_composite.putScalar(0.0);
    dgdp_combined.putScalar(0.0);
    composite.computeDP(paramIDs, dgdp_composite, false);
    combined.computeDP(paramIDs, dgdp_combined, false);

    ierr += testCompare.testMatrix(
                 dgdp_composite, dgdp_combined, reltol, abstol,
                 "CompositeConstraint::computeDP()");

    //
    // test multiplyDX()
    //

    composite.computeDX();
    combined.computeDX();

    int numMultiply = 5;
    Teuchos::RCP<NOX::Abstract::MultiVector> A =
      nox_clone_vec.createMultiVector(numMultiply);
    A->random();
    NOX::Abstract::MultiVector::DenseMatrix composite_multiply(numConstraints,
                                   numMultiply);
    NOX::Abstract::MultiVector::DenseMatrix combined_multiply(numConstraints,
                                  numMultiply);
    composite.multiplyDX(2.65, *A, composite_multiply);
    combined.multiplyDX(2.65, *A, combined_multiply);

    ierr += testCompare.testMatrix(composite_multiply, combined_multiply,
                    reltol, abstol,
                    "CompositeConstraint::multiplyDX()");

    //
    // test addDX() (No Trans)
    //

    int numAdd = 5;
    NOX::Abstract::MultiVector::DenseMatrix B1(numConstraints, numAdd);
    B1.random();
    NOX::Abstract::MultiVector::DenseMatrix B2(numAdd, numConstraints);
    B2.random();

    Teuchos::RCP<NOX::Abstract::MultiVector> composite_add1 =
      nox_clone_vec.createMultiVector(numAdd);
    composite_add1->random();
    Teuchos::RCP<NOX::Abstract::MultiVector> composite_add2 =
      nox_clone_vec.createMultiVector(numAdd);
    composite_add2->random();

    Teuchos::RCP<NOX::Abstract::MultiVector> combined_add1 =
      composite_add1->clone(NOX::DeepCopy);
    Teuchos::RCP<NOX::Abstract::MultiVector> combined_add2 =
      composite_add2->clone(NOX::DeepCopy);

    composite.addDX(Teuchos::NO_TRANS, 1.45, B1, 2.78, *composite_add1);
    combined.addDX(Teuchos::NO_TRANS, 1.45, B1, 2.78, *combined_add1);

    ierr += testCompare.testMultiVector(
                   *composite_add1, *combined_add1,
                   reltol, abstol,
                   "CompositeConstraint::addDX() (No Trans)");

    //
    // test addDX() (Trans)
    //

    composite.addDX(Teuchos::TRANS, 1.45, B2, 2.78, *composite_add2);
    combined.addDX(Teuchos::TRANS, 1.45, B2, 2.78, *combined_add2);

    ierr += testCompare.testMultiVector(
                   *composite_add2, *combined_add2,
                   reltol, abstol,
                   "CompositeConstraint::addDX() (Trans)");

    LOCA::destroyGlobalData(globalData);
  }

  catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    ierr = 1;
  }
  catch (const char *s) {
    std::cout << s << std::endl;
    ierr = 1;
  }
  catch (...) {
    std::cout << "Caught unknown exception!" << std::endl;
    ierr = 1;
  }

  if (MyPID == 0) {
    if (ierr == 0)
      std::cout << "All tests passed!" << std::endl;
    else
      std::cout << ierr << " test(s) failed!" << std::endl;
  }

  return ierr;
}

int main(int argc, char *argv[]) {
  return run<double>(argc,argv);
  // return run<float>(argc,argv);
}
