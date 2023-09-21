/** \file shylu_belos_driver_tpetra.cpp

    \brief Factors and solves a sparse matrix using LU factorization.

    \author Caleb Schilly (adapted from shylu_belos_driver.cpp by Siva Rajamanickam)

    \remark Usage:
    \code mpirun -n np shylu_belos_driver_tpetra.exe

*/

#include <sstream>
#include <iostream>
#include <assert.h>

// Tpetra includes
#include <Tpetra_Map.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <MatrixMarket_Tpetra.hpp>

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

// MueLu
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

#include "shylu.h"
#include "shylu_util.h"
#include "ShyLU_DDCore_config.h"
#include "shylu_partition_interface.hpp"
#include "shylu_directsolver_interface.hpp"


// Will likely have to move somewhere
int InitMatValues( const tcrsmatrix_t& newA, tcrsmatrix_t* A );
int InitMVValues( const MV& newb, MV* b );


template <typename ScalarType>
int run(int argc, char *argv[]) {

  using ST  = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO  = typename Tpetra::MultiVector<>::local_ordinal_type;
  using GO  = typename Tpetra::MultiVector<>::global_ordinal_type;
  using NT  = typename Tpetra::MultiVector<>::node_type;

  using SCT = typename Teuchos::ScalarTraits<ST>;
  using MT  = typename SCT::magnitudeType;
  using MV  = typename Tpetra::MultiVector<ST,LO,GO,NT>;
  using OP  = typename Tpetra::Operator<ST,LO,GO,NT>;
  using MOP = typename MueLu::TpetraOperator<ST,LO,GO,NT>

  using MVT = Belos::MultiVecTraits<ST,MV>;
  using OPT = Belos::OperatorTraits<ST,MV,OP>;

  using tmap_t = Tpetra::Map<LO,GO,NT>;
  using tcrsmatrix_t = Tpetra::CrsMatrix<ST,LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);
  const auto Comm = Tpetra::getDefaultComm();

  bool success = true;
  std::string pass = "End Result: TEST PASSED";
  std::string fail = "End Result: TEST FAILED";

  bool verbose = false, proc_verbose = true;
  bool leftprec = false;      // left preconditioning or right.
  int frequency = -1;         // frequency of status test output.
  int blocksize = 1;          // blocksize
  int numrhs = 1;             // number of right-hand sides to solve for
  int maxrestarts = 15;       // maximum number of restarts allowed
  int maxsubspace = 25;       // maximum number of blocks the solver can use
                              // for the subspace
  char file_name[100];

  int nProcs, myPID ;
  Teuchos::RCP <Teuchos::ParameterList> pLUList ;        // ParaLU parameters
  Teuchos::ParameterList z2List ;        // Isorropia parameters
  Teuchos::ParameterList shyLUList ;    // ShyLU parameters
  std::string ipFileName = "ShyLU.xml";       // TODO : Accept as i/p

  nProcs = Comm->getSize();
  myPID = Comm->getRank();

  if (myPID == 0) {
    std::cout <<"Parallel execution: nProcs="<< nProcs << std::endl;
  }

  // =================== Read input xml file =============================
  pLUList = Teuchos::getParametersFromXmlFile(ipFileName);
  z2List = pLUList->sublist("Zoltan2 Input");
  shyLUList = pLUList->sublist("ShyLU Input");
  shyLUList.set("Outer Solver Library", "Belos");
  // Get matrix market file name
  std::string MMFileName = Teuchos::getParameter<std::string>(*pLUList, "mm_file");
  std::string prec_type = Teuchos::getParameter<std::string>(*pLUList, "preconditioner");
  int maxiters = Teuchos::getParameter<int>(*pLUList, "Outer Solver MaxIters");
  MT tol = Teuchos::getParameter<double>(*pLUList, "Outer Solver Tolerance");
  std::string rhsFileName = pLUList->get<std::string>("rhs_file", "");


  int maxFiles = pLUList->get<int>("Maximum number of files to read in", 1);
  int startFile = pLUList->get<int>("Number of initial file", 1);
  int file_number = startFile;

  if (myPID == 0) {
    std::cout << "Input :" << std::endl;
    std::cout << "ParaLU params " << std::endl;
    pLUList->print(std::cout, 2, true, true);
    std::cout << "Matrix market file name: " << MMFileName << std::endl;
  }

  if (maxFiles > 1) {
    MMFileName += "%d.mm";
    sprintf( file_name, MMFileName.c_str(), file_number );
  } else {
    strcpy( file_name, MMFileName.c_str());
  }

  // ==================== Read input Matrix ==============================
  RCP<tcrsmatrix_t> A;
  RCP<MV> b1;

  A = Tpetra::MatrixMarket::Reader<tcrsmatrix_t>::readSparseFile(file_name, Comm);

  int n = A->getNumGlobalRows();

  // ==================== Read input rhs  ==============================
  if (rhsFileName != "" && maxFiles > 1) {
    rhsFileName += "%d.mm";
    sprintf( file_name, rhsFileName.c_str(), file_number );
  } else {
    strcpy( file_name, rhsFileName.c_str());
  }

  tmap_t vecMap(n, 0, Comm);
  bool allOneRHS = false;
  if (rhsFileName != "") {
    b1 = Tpetra::MatrixMarket::Reader<tcrsmatrix_t>::readSparseFile(file_name, vecMap);
  } else {
    b1 = new rcp<MV>(vecMap, 1, false);
    b1->random();
    allOneRHS = true;
  }

  MV x(vecMap, 1);

  // Partition the matrix with hypergraph partitioning and redisstribute
  ShyLU::PartitionInterface<Matrix_t, Vector_t> partitioner(A.get(), pLUList.get());

  Teuchos::Time ptime("Partition time");
  ptime.start();
  partitioner.partition();
  ptime.stop();

  if (myPID == 0) {
    std::cout << "Time to partition   : " << ptime.totalElapsedTime() << std::endl << std::endl;
  }

  RCP<tcrsmatrix_t> newA;
  RCP<MV> newX, newB;

  A = newA;

  RCP<tcrsmatrix_t> rcpA(A, false);

  RCP<MV> rcpx (newX, false);
  RCP<MV> rcpb (newB, false);

  if (myPID == 0) {
    std::cout << "Time to redistribute: " << rtime.totalElapsedTime() << std::endl << std::endl;
  }

  RCP<tcrsmatrix_t> iterA = 0;
  RCP<tcrsmatrix_t> redistA = 0;
  RCP<MV> iterb1 = 0;

  // Ifpack_Preconditioner *prec;

  while(file_number < maxFiles+startFile) {

      if (prec_type.compare("ShyLU") == 0)
      {
          if (file_number == startFile)
          {
              Teuchos::Time itime("Initialize time");
              itime.start();
              prec = new Ifpack_ShyLU(A);
#ifdef HAVE_IFPACK_DYNAMIC_FACTORY
              Teuchos::ParameterList shyluParameters;
              shyluParameters.set<Teuchos::ParameterList>("ShyLU list", shyLUList);
              prec->SetParameters(shyluParameters);
#else
              prec->SetParameters(shyLUList);
#endif
              prec->Initialize();
              itime.stop();
              if (myPID == 0)
              {
                  std::cout << "Time to initialiize : " << itime.totalElapsedTime() << std::endl << std::endl;
              }
          }
          Teuchos::Time ftime("Compute    time");
          ftime.start();
          prec->Compute();
          ftime.stop();
          if (myPID == 0)
          {
              std::cout << "Time to compute     : " << ftime.totalElapsedTime() << std::endl << std::endl;
          }
          //std::cout << " Going to set it in solver" << std::endl ;
          //solver.SetPrecOperator(prec);
          //std::cout << " Done setting the solver" << std::endl ;
      }
      else if (prec_type.compare("ILU") == 0)
      {
          prec = new Ifpack_ILU(A);
          prec->Initialize();
          prec->Compute();
          //solver.SetPrecOperator(prec);
      }
      else if (prec_type.compare("ILUT") == 0)
      {
          prec = new Ifpack_ILUT(A);
          prec->Initialize();
          prec->Compute();
          //solver.SetPrecOperator(prec);
      }
      else if (prec_type.compare("MueLu") == 0)
      {
          // Teuchos::ParameterList mueluList; // CWS: think replaced by ShyLUList

          RCP<OP> A_op = A;
          RCP<mtoperator_t> Prec = MueLu::CreateTpetraPreconditioner(A_op, shyLUList);
      }

      RCP<Ifpack_Preconditioner> rcpPrec(prec, false);

      const int NumGlobalElements = rcpb->GlobalLength();
      Teuchos::ParameterList belosList;

      belosList.set( "Num Blocks", maxsubspace );// Maximum number of blocks in Krylov factorization
      belosList.set( "Block Size", blocksize );  // Blocksize to be used by iterative solver
      belosList.set( "Maximum Iterations", maxiters ); // Maximum number of iterations allowed
      belosList.set( "Maximum Restarts", maxrestarts );// Maximum number of restarts allowed
      belosList.set( "Convergence Tolerance", tol );   // Relative convergence tolerance requested

      if (numrhs > 1) {
        belosList.set( "Show Maximum Residual Norm Only", true );  // Show only the maximum residual norm
      }
      if (verbose) {
        belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
              Belos::TimingDetails + Belos::StatusTestDetails );
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
      }
      else
      belosList.set( "Verbosity", Belos::Errors + Belos::Warnings );

      //
      // *******Construct a preconditioned linear problem********
      //

      rcpx->putScalar(0.0);

      RCP<Belos::LinearProblem<double,MV,OP> > problem
          = rcp( new Belos::LinearProblem<double,MV,OP>( rcpA, rcpx, rcpb ) );

      if (leftprec) {
          problem->setLeftPrec( Prec );
      }
      else {
          problem->setRightPrec( Prec );
      }

      bool set = problem->setProblem();
      if (set == false) {
          if (proc_verbose)
          {
              std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
          }
          std::cout << fail << std::endl;
          success = false;
          return -1;
      }

      // Create an iterative solver manager.
      RCP< Belos::SolverManager<double,MV,OP> > solver
        = rcp( new Belos::BlockGmresSolMgr<double,MV,OP>(
              problem,
              rcp(&belosList,false)
            )
          );

      //
      // *******************************************************************
      // *************Start the block Gmres iteration*************************
      // *******************************************************************
      //

      // CWS: pick up here

      if (proc_verbose && myPID == 0)
      {
          std::cout << std::endl << std::endl;
          std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
          std::cout << "Number of right-hand sides: " << numrhs << std::endl;
          std::cout << "Block size used by solver: " << blocksize << std::endl;
          std::cout << "Number of restarts allowed: " << maxrestarts << std::endl;
          std::cout << "Max number of Gmres iterations per restart cycle: " <<
                      maxiters << std::endl;
          std::cout << "Relative residual tolerance: " << tol << std::endl;
          std::cout << std::endl;
      }

      if(tol > 1e-5)
        {
          success = false;
        }

      //
      // Perform solve
      //
      Teuchos::Time stime("Solve      time");
      stime.start();
      solver->solve ();
      stime.stop();
      if (myPID == 0)
      {
          std::cout << "Time to solve       : " << stime.totalElapsedTime() << std::endl << std::endl;
      }

      //
      // Get the number of iterations for this solve.
      //
      int numIters = solver->getNumIters();
      if (proc_verbose && myPID == 0)
      {
          std::cout << "Number of iterations performed for this solve: " <<
                    numIters << std::endl;
      }
      //
      // Compute actual residuals.
      //
      //bool badRes = false; // unused
      std::vector<double> actual_resids( numrhs );
      std::vector<double> rhs_norm( numrhs );
      MV resid((*rcpA).RowMap(), numrhs);
      OPT::Apply( *rcpA, *rcpx, resid );
      MVT::MvAddMv( -1.0, resid, 1.0, *rcpb, resid );
      MVT::MvNorm( resid, actual_resids );
      MVT::MvNorm( *rcpb, rhs_norm );
      if (proc_verbose && myPID == 0)
      {
          std::cout<< "------ Actual Residuals (normalized) -------"<<std::endl;
          for ( int i=0; i<numrhs; i++)
          {
              double actRes = actual_resids[i]/rhs_norm[i];
              std::cout<<"Problem "<<i<<" : \t"<< actRes;
              if (actRes > tol) {
                //badRes = true; // unused
                std::cout<<" (NOT CONVERGED)"<< std::endl;
                success = false;
              } else {
                std::cout<<" (CONVERGED)"<< std::endl;
              }
          }
      }

      file_number++;
      if (file_number >= maxFiles+startFile)
      {
        break;
      }
      else
      {
          sprintf(file_name, MMFileName.c_str(), file_number);

          if (redistA != NULL) delete redistA;
          // Load the new matrix
          iterA = EpetraExt::MatrixMarketFileToCrsMatrix(file_name,
                          Comm);
          if (err != 0)
          {
              if (myPID == 0)
                {
                  std::cout << "Could not open file: "<< file_name << std::endl;

                }
              success = false;
          }
          else
          {
              rd.redistribute(*iterA, redistA);
              delete iterA;
              InitMatValues(*redistA, A);
          }

          // Load the new rhs
          if (!allOneRHS)
          {
              sprintf(file_name, rhsFileName.c_str(), file_number);

              if (iterb1 != NULL) delete iterb1;
              err = EpetraExt::MatrixMarketFileToMultiVector(file_name,
                      vecMap, b1);
              if (err != 0)
              {
                  if (myPID==0)
                    {
                      std::cout << "Could not open file: "<< file_name << std::endl;
                      success = false;
                    }
              }
              else
              {
                  rd.redistribute(*b1, iterb1);
                  delete b1;
                  InitMVValues( *iterb1, newB );
              }
          }
      }
  }
  if (myPID == 0)
  {
      if(success)
        {
          std::cout << pass << std::endl;
        }
      else
        {
          std::cout << fail << std::endl;
        }
  }

  if (redistA != NULL) delete redistA;
  if (iterb1 != NULL) delete iterb1;


  if (prec_type.compare("ML") == 0)
  {
      delete MLprec;
  }
  else
  {
      delete prec;
  }
  delete newX;
  delete newB;
  delete A;
  delete partitioner;
}

int InitMatValues( const tcrsmatrix_t& newA, tcrsmatrix_t* A )
{
  int numMyRows = newA.NumMyRows();
  int maxNum = newA.MaxNumEntries();
  int numIn;
  int *idx = 0;
  double *vals = 0;

  idx = new int[maxNum];
  vals = new double[maxNum];

  // For each row get the values and indices, and replace the values in A.
  for (int i=0; i<numMyRows; ++i) {

    // Get the values and indices from newA.
    EPETRA_CHK_ERR( newA.ExtractMyRowCopy(i, maxNum, numIn, vals, idx) );

    // Replace the values in A
    EPETRA_CHK_ERR( A->ReplaceMyValues(i, numIn, vals, idx) );

  }

  // Clean up.
  delete [] idx;
  delete [] vals;

  return 0;
}

int InitMVValues( const MV& newb, MV* b )
{
  int length = newb.MyLength();
  int numVecs = newb.NumVectors();
  const tvector_t *tempnewvec;
  tvector_t *tempvec = 0;

  for (int i=0; i<numVecs; ++i) {
    tempnewvec = newb(i);
    tempvec = (*b)(i);
    for (int j=0; j<length; ++j)
      (*tempvec)[j] = (*tempnewvec)[j];
  }

  return 0;
}
