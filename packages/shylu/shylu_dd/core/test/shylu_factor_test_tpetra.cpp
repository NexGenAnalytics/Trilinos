/** \file shylu_sfactor_test_tpetra.cpp

    \brief factor test

    \author Caleb Schilly (adapted from shylu_factor_test.cpp by Joshua Dennis Booth)

    \remark Usage:
    \code mpirun -n np shylu_sfactor_tpetra.exe

*/
// Will be used to test gFACT as we morph into templated

// #ifdef HAVE_SHYLU_DDCORE_TPETRA

#include <assert.h>
#include <iostream>
#include <sstream>

#include "Isorropia_config.h" // Just for HAVE_MPI

// #include "Epetra_LinearProblem.h"

// Tpetra includes
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <MatrixMarket_Tpetra.hpp>

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_RCP.hpp"

#include "shylu.h"
#include "shylu_util.h"

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

  using MVT = typename Belos::MultiVecTraits<ST,MV>;

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
  const auto Comm = Tpetra::getDefaultComm();

  string pass = "End Result: TEST PASSED";
  string fail = "End Result: TEST FAILED";

  int myPID = Comm->getRank();

  if(myPID == 0)
    {
      cout << "Starting SFactor Epetra Test" << endl;
    }

  //============================= Get Matrix
  string matrixFileName = "wathenSmall.mtx";

  RCP<tcrsmatrix_t> A = Tpetra::MatrixMarket::Reader<tcrsmatrix_t>::readSparseFile(matrixFileName, Comm);

  if(err!=0 && myPID ==0)
    {
      cout << "Error reading matrix file, info = " << err << endl;
      cout << fail << endl;
      exit(1);
    }

  //=============================Partition/Distribute
  Teuchos::ParameterList isoList;

  cout << "before partition" << endl;
  tcrsmatrix_t *B = balanceAndRedistribute(A,isoList);
  cout << "after partition" << endl;

  shylu_data     slu_data_;
  shylu_config   slu_config_;
  shylu_symbolic slu_sym_;

  slu_config_.sym                 = 1;     //This is
  slu_config_.libName             = "Belos"; //This is
  slu_config_.schurSolver         = " "; //This is
  slu_config_.schurAmesosSolver   = " " ; //This is
  slu_config_.diagonalBlockSolver = "Amesos2_Klu"; //This is
  slu_config_.relative_threshold  = 0.0; //This is
  slu_config_.Sdiagfactor         = 0.05; //This is
  slu_config_.schurApproxMethod   = 2;  //1 A22Block  2 Thresholding 3 Guided
  slu_config_.schurPreconditioner = "ILU stand-alone"; //This is
  slu_config_.silent_subiter      = true; //This is
  slu_config_.inner_tolerance     = 1e-5; //This is
  slu_config_.inner_maxiters      = 100; //This is
  slu_config_.overlap             = 0; //This is
  slu_config_.sep_type            = 1; //1 Wide 2 Narrow
  slu_config_.amesosForDiagonal   = true;


  int serr = shylu_symbolic_factor(B, &slu_sym_, &slu_data_, &slu_config_);
  cout << "shylu_symbolic_factor done:" << endl;
  cout << "Return value: " << serr << endl;
  if(serr == 0)
    cout << pass << endl;
  return serr;
} //end run

int main(int argc, char *argv[]) {
  // allow for running multiple ST
  run<double>(argc,argv);
}

// #endif // HAVE_SHYLU_DDCORE_TPETRA