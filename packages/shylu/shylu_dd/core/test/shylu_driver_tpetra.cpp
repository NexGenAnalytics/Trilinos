/** \file shylu_driver_tpetra.cpp

    \brief Factors and solves a sparse matrix using LU factorization.

    \author Caleb Schilly (adapted from shylu_driver.cpp by Siva Rajamanickam)

    \remark Usage:
    \code mpirun -n np shylu_driver_tpetra.exe

*/

#include <assert.h>
#include <iostream>
#include <sstream>

#include "Isorropia_config.h" // Just for HAVE_MPI

#include <Ifpack2_Partitioner.hpp>

// Tpetra includes
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <MatrixMarket_Tpetra.hpp>

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

// Amesos2 includes
#include <Amesos2.hpp>

// MueLu includes
#include <MueLuPreconditioner.hpp>

#include "shylu.h"
#include "shylu_util.h"
#include "ShyLU_DDCore_config.h"
#include "shylu_partition_interface.hpp"
#include "shylu_directsolver_interface.hpp"


using namespace std;

template <typename ScalarType>
int run(int argc, char *argv[]) {

  using ST = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO = typename Tpetra::MultiVector<>::local_ordinal_type;
  using GO = typename Tpetra::MultiVector<>::global_ordinal_type;
  using NT = typename Tpetra::MultiVector<>::node_type;

  using SCT = typename Teuchos::ScalarTraits<ST>;
  using MT  = typename SCT::magnitudeType;
  using MV  = typename Tpetra::MultiVector<ST,LO,GO,NT>;

  using tmap_t = Tpetra::Map<LO,GO,NT>;
  using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
  const auto Comm = Tpetra::getDefaultComm();

  int nProcs, myPID ;
  Teuchos::ParameterList pLUList ;        // ParaLU parameters
  Teuchos::ParameterList shyLUList ;      // shyLU parameters
  Teuchos::ParameterList ifpackList ;     // shyLU parameters
  string ipFileName = "ShyLU.xml";        // TODO : Accept as i/p

  nProcs = Comm->getSize();
  myPID = Comm->getRank();

  if (myPID == 0) {
    cout <<"Parallel execution: nProcs="<< nProcs << endl;
  }

  // =================== Read input xml file =============================

  Teuchos::updateParametersFromXmlFile(ipFileName, &pLUList);
  shyLUList = pLUList.sublist("ShyLU Input");
  shyLUList.set("Outer Solver Library", "Amesos2");

  // Get matrix market file name
  string MMFileName = Teuchos::getParameter<string>(pLUList, "mm_file");

  string prec_type = Teuchos::getParameter<string>(pLUList, "preconditioner");
  int maxiters = Teuchos::getParameter<int>(pLUList, "Outer Solver MaxIters");
  double tol = Teuchos::getParameter<double>(pLUList, "Outer Solver Tolerance");
  string rhsFileName = pLUList.get<string>("rhs_file", "");

  // Partitioning
  pLUList->set("Partitioning Package","Zoltan2");
  Teuchos::ParameterList ptemp = pLUList->sublist("Zoltan2 Input");
  ptemp.set("algorithm", "parmetis");
  ptemp.set("debug_level", "detailed_status");
  pLUList->set("Zoltan2 Input", ptemp);

  if (myPID == 0) {
    cout << "Input :" << endl;
    cout << "ParaLU params " << endl;
    pLUList.print(std::cout, 2, true, true);
    cout << "Matrix market file name: " << MMFileName << endl;
  }

  // ==================== Read input Matrix ==============================
  RCP<tcrsmatrix_t> A;
  RCP<tmultivector_t> b1;

  A = Tpetra::MatrixMarket::Reader<tcrsmatrix_t>::readSparseFile(MMFileName, Comm);

  int n = A->getNumGlobalRows();

  // Create input vectors
  tmap_t vecMap(n, 0, Comm);

  if (rhsFileName != "") {
    b1 = Tpetra::MatrixMarket::Reader<tcrsmatrix_t>::readSparseFile(rhsFileName,vecMap);
  } else {
    b1 = (new rcp<tmultivector_t>(vecMap, 1, false));
    b1->putScalar(1.0);
  }

  tmultivector_t x(vecMap, 1);

  // Partition the matrix with hypergraph partitioning and redistribute
#ifdef HAVE_SHYLU_DDCORE_ZOLTAN2

  ShyLU::PartitionInterface<Matrix_t, Vector_t> partitioner(A.get(), pLUList.get());
  partitioner.partition();

  if(myPID == 0) {
    cout << "Done with graph - parmetis" << endl;
  }

#else

  success = false;

#endif

  RCP<tcrsmatrix_t> newA;
  RCP<tmultivector_t> newX, newB;
  A = newA;

#ifdef HAVE_SHYLU_DDCORE_AMESOS2

  pLUList->set("Direct Solver Package", "Amesos2");
  ptemp = pLUList->sublist("Amesos2 Input");

  ptemp.set("Solver", "SuperLU");
  pLUList->set("Amesos2 Input", ptemp);

  if(myPID == 0) {
    cout << "\n\n--------------------BIG BREAK --------------\n\n";
    Teuchos::writeParameterListToXmlOStream(*pLUList, std::cout);

    cout << "num_vector: " << newB->getNumVectors() << " "
          << newX->getNumVectors() << endl;
    cout << "length: " << newB->getGlobalLength() << " "
          << newX->getGlobalLength() << endl;

    cout << "A length" << A->getGlobalNumRows() << " " << A->getGlobalNumCols() << endl;
    cout << "A local length" << A->getLocalNumRows() << " " << A->getLocalNumCols() << endl;
  }

  ShyLU::DirectSolverInterface<Matrix_t, Vector_t> directsolver2(A.get(), pLUList.get());
  directsolver2.factor();
  directsolver2.solve(newB.get(),newX.get());

  if(myPID == 0) {
    cout << "Done with Amesos2-SuperLU" << endl;
  }

//Note: should multiple to set b and x for success

#else

  success = false;

#endif // HAVE_SHYLU_DDCORE_AMESOS2

  ifpack2List ;
  Ifpack2::Preconditioner *prec;
  MueLu::Preconditioner *MLprec;
  if (prec_type.compare("ShyLU") == 0)
  {
      prec = new Ifpack2_ShyLU(A);
      prec->SetParameters(shyLUList);
      prec->Initialize();
      prec->Compute();
      //(dynamic_cast<Ifpack_ShyLU *>(prec))->JustTryIt();
      //cout << " Going to set it in solver" << endl ;
      solver.SetPrecOperator(prec);
      //cout << " Done setting the solver" << endl ;
  }
  else if (prec_type.compare("ILU") == 0)
  {
      ifpackList.set( "fact: level-of-fill", 1 );
      prec = new Ifpack_ILU(A);
      prec->SetParameters(ifpackList);
      prec->Initialize();
      prec->Compute();
      solver.SetPrecOperator(prec);
  }
  else if (prec_type.compare("ILUT") == 0)
  {
      ifpackList.set( "fact: ilut level-of-fill", 2 );
      ifpackList.set( "fact: drop tolerance", 1e-8);
      prec = new Ifpack_ILUT(A);
      prec->SetParameters(ifpackList);
      prec->Initialize();
      prec->Compute();
      solver.SetPrecOperator(prec);
  }
  else if (prec_type.compare("ML") == 0)
  {
      Teuchos::ParameterList mlList; // TODO : Take it from i/p
      MLprec = new ML_Epetra::MultiLevelPreconditioner(*A, mlList, true);
      solver.SetPrecOperator(MLprec);
  }

  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetMatrixName(333);
  //solver.SetAztecOption(AZ_output, 1);
  //solver.SetAztecOption(AZ_conv, AZ_Anorm);
  //cout << "Going to iterate for the global problem" << endl;

  solver.Iterate(maxiters, tol);

  // compute ||Ax - b||
  double Norm;
  tmultivector_t Ax(vecMap, 1);

  tmultivector_t *newAx;
  rd.redistribute(Ax, newAx);
  A->Multiply(false, *newX, *newAx);
  newAx->Update(1.0, *newB, -1.0);
  newAx->Norm2(&Norm);
  double ANorm = A->NormOne();

  cout << "|Ax-b |/|A| = " << Norm/ANorm << endl;

}

int main(int argc, char *argv[]) {
  // allow for running multiple ST
  run<double>(argc,argv);
}
