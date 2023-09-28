/** \file shylu_iqr_driver.cpp

    \brief Factors and solves a sparse matrix using ShyLU and IQR, if enabled.

    \author Radu Popescu <i.radu.popescu@gmail.com>

    * The example needs the following Ifpack variables enabled when compiling Trilinos:
      -DIfpack_ENABLE_DYNAMIC_FACTORY:BOOL=ON (allows the use of the new factory which can register
      new preconditioners within the client application)
      -DIfpack_ENABLE_PARALLEL_SUBDOMAIN_SOLVERS:BOOL=ON (allows using MPI parallel subdomain
      solvers for Ifpack AAS)
    * The example doesn't fail if these variables are disabled, but it doesn't do anything
    * interesting, it just solves the problem with Ifpack AAS with serial Amesos subdomain solvers.

    * Highlights:
      * There is a builder function for Ifpack_ShyLU declared at the beginning of the file.
      * Ifpack_ShyLU is registered with the Ifpack_DynamicFactory class at the beginning of the
        main() function, using the builder function that was declared earlier.
      * An alternate XML file is provided in case the CMake variables mentioned earlier are not
        enabled, to prevent the test from failing.
      * A global Teuchos parameter list is read from an XML file, with sublists for the matrix
        partitioner, the preconditioner and the linear solver
      * The ML list contains a sublist for Ifpack (as usual), which in turn contains a sublist for
          all ShyLU parameters (this behaviour is enabled with the Ifpack_DynamicFactory switch, to
          avoid parameter list polution)
      * Through Ifpack_DynamicFactory, ML is be able to build the Ifpack_AdditiveSchwarz<Ifpack_ShyLU>
        preconditioner.

    * Key parameters:
      * Isorropia parameters:
        * we set the partitioning method to HIER_GRAPH, to ensure that the parts which make up each
          AAS subdomain are connected
        * we set TOPOLOGY to the number of processor per AAS subdomain
      * ML parameters:
        * smoother: ifpack type - string - should be set to ShyLU
        * smoother: ifpack overlap - int - 0 - using multiple processors per AAS subdomain forces
          this. Overlap is not supported.
      * Ifpack parameters:
        * subdomain: number-of-processors - int - number of processors per AAS subdomain (must be a
          divisor of the total number of MPI processes used).
      * ShyLU parameters:
        * Schur Approximation Method - usually A22AndBlockDiagonals; can be set to IQR or G. IQR
          means that we use IQR to solve the Schur complement system inexactly (Krylov subspace
          reuse method), G means that we just approximate the inverse of the Schur complement with
          an AAS of the G subblock of the subdomain matrix (which is faster, but leads to a looser
          coupling at the subdomain level, for some problems it leads to fewer outer GMRES iterations
          than IQR).
        * IQR Initial Prec Type - string - default: Amesos (this is the actual string given to an
          Ifpack factory; it means AAS with serial Amesos on subdomains) - the preconditioner used
          for the GMRES solver within IQR. This is also used for the approximation of the Schur
          complement inverse when Schur Approximation Method is set to G.
        * IQR Initial Prec Amesos Type - string - default: Amesos_Klu - which Amesos solver to use
          for the IQR preconditioner
      * Amesos_Klu is given as a default in multiple places of the XML file. Should be substituted
        with faster alternatives: Amesos_Pardiso, Amesos_Umfpack etc.

    \remark Usage:
    \code mpirun -n 2np ShyLU_iqr_driver.exe

*/

// Standard
#include <string>


#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// Tpetra
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_BlockCrsMatrix.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <MatrixMarket_Tpetra.hpp>  // MatrixMarket File I/O

// #include <EpetraExt_CrsMatrixIn.h>
// #include "EpetraExt_MultiVectorIn.h"
// #include "EpetraExt_BlockMapIn.h"

// Belos
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// #include <Ifpack_ConfigDefs.h>
// #if defined(HAVE_IFPACK_DYNAMIC_FACTORY) && defined(HAVE_IFPACK_PARALLEL_SUBDOMAIN_SOLVERS)
// #if defined(HAVE_SHYLU_IFPACK2)
// #include <Ifpack_DynamicFactory.h>
// #include <Ifpack_AdditiveSchwarz.h>
// #include <Ifpack_ShyLU.h>
// #else
// #include <Ifpack.h>
// #endif

#include <Ifpack2_AdditiveSchwarz.hpp>
#include <Ifpack2_ConfigDefs.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Parameters.hpp>
#include <Ifpack2_Preconditioner.hpp>

// #include <Isorropia_EpetraPartitioner.hpp> // TD: No Isorropia Tpetra alternative found
// #include <Isorropia_EpetraRedistributor.hpp> // TD: No Isorropia Tpetra alternative found
#include <Ifpack2_Partitioner.hpp>  // TD: could this replace Isorropia_EpetraPartitioner ?
#include <Tpetra_Distributor.hpp>   // TD: could this replace replace Isorropia_EpetraRedistributor ?

// #include <ml_MultiLevelPreconditioner.h> // TD: this is Epetra. Replace by Ifpack2_Preconditioner ?

// TD: `#if defined(HAVE_IFPACK_DYNAMIC_FACTORY) && defined(HAVE_IFPACK_PARALLEL_SUBDOMAIN_SOLVERS)`
// TD:  becomes:
// TD:  `if (useIqr)` and will be passed as CLI argument (use-iqr, default true)

using Teuchos::ParameterList;
using Teuchos::RCP;
using Teuchos::rcp;

namespace {

template <class CrsMatrix>
int initMatValues(const CrsMatrix &srcMatrix, const CrsMatrix<> *dstMatrix) {
  using ST = CrsMatrix::scalar_type;
  using LO = CrsMatrix::local_ordinal_type;

  using indsView = typename CrsMatrix::local_inds_host_view_type;
  using valsView = typename CrsMatrix::values_host_view_type;

  int numRows = newA.getLocalNumRows();
  int maxNumRowEntries = newA.getLocalMaxNumRowEntries();
  int numEntries = 0;

  LO *indices = new LO[maxNum];
  ST *values = new ST[maxNum];

  // For each row get the values and indices, and replace the values in A.
  for (size_t i = 0; i < numRows; ++i) {
    // Get the values and indices from newA.
    indsView indicesView(indices, maxNumRowEntries);
    valsView valuesView(values, maxNumRowEntries);
    srcMatrix->getLocalRowCopy(i, indicesView, valuesView, numEntries);

    // Replace the values in A
    dstMatrix->replaceLocalValues(i, indices, values, numEntries);
  }

  // Clean up.
  delete[] indices;
  delete[] values;

  return 0;
}

template <class MultiVector>
int initMVValues(const MultiVector &srcVec, MultiVector *dstVec) {
  using Vector = Tpetra::Vector<>;

  int length = srcVec.getLocalLength();
  int numVecs = srcVec.getNumVectors();
  const Vector *tempnewvec;
  Vector *tempvec = 0;

  for (int i = 0; i < numVecs; ++i) {
    tempnewvec = srcVec(i);
    tempvec = (*dstVec)(i);
    for (int j = 0; j < length; ++j) (*tempvec)[j] = (*tempnewvec)[j];
  }

  return 0;
}

template <typename Scalar>
int run(int argc, char **argv) {
  using ST = typename Tpetra::Vector<Scalar>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  using MultiVector = typename Tpetra::MultiVector<ST, LO, GO, NT>;
  using Operator = typename Tpetra::Operator<ST, LO, GO, NT>;

  using RowMatrix = typename Tpetra::RowMatrix<ST, LO, GO, NT>;
  using CrsMatrix = typename Tpetra::CrsMatrix<ST, LO, GO, NT>;
  using BlockCrsMatrix = typename Tpetra::BlockCrsMatrix<ST, LO, GO, NT>;
  using Map = typename Tpetra::Map<LO, GO, NT>;
  using RowGraph = Tpetra::RowGraph<LO, GO, NT>;

  using Preconditioner = typename Ifpack2::Preconditioner<ST, LO, GO, NT>;

  bool success = true;

  try
  {
    // Initialize MPI environment
    Teuchos::GlobalMPISession mpiSession(&argc, &argv);
    RCP<const comm<int> > comm = Tpetra::getDefaultComm();

    const int myRank = comm->getRank();
    const int numProcs = comm->getSize();
    bool verbose = (myRank == 0);

    if (verbose) {
      std::cout << "--- Running ShyLU-IQR test with " << nProcs << " processes." << std::endl;
    }

    std::string parameterFileName = "";
    bool useIqr = true;
    
    // Accept an alternate XML parameter file name, read from the command line
    Teuchos::CommandLineProcessor cmdp(false, false);
    cmdp.setDocString("");
    cmdp.setOption("parameter-file", &parameterFileName, "The name of the XML test parameter file");
    cmdp.setOption("use-iqr", "no-iqr", &useIqr, "Whether to use iqr parameter file by default");
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = cmdp.parse(argc, argv);

    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      std::cout << fail << std::endl;
      return 0;
    }

    if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      std::cout << fail << std::endl;
      return -2;
    }

    // set default parameter file for
    // the partitioner, the preconditioner, linear solver etc.
    // TODO: set correct parameters in the xml files adapted to this tpetra test
    if (parameterFileName == "") {
      if (useIqr) {
        parameterFileName = "shylu_iqr_parameters_tpetra.xml";
      } else {
        parameterFileName = "shylu_no_iqr_parameters_tpetra.xml";
      }
    }

    if (verbose) {
      std::cout << "--- Using parameter file: " << parameterFileName << std::endl;
    }

    // TD: Epetra: no RegisterPreconditioner alternative found
    // Seems that "ShyLU" preconditioner was a copy of Schwartz preconditioner with the name "ShyLU"
    // Create ShyLU preconditioner as an AdditiveSchwartz preconditioner
    // TD: try the following. Should be inherit the Swhwarz preconditioner ?
    // if (useIqr) {
    //     // Epetra: previously done by buildShyLU method
    //     // Epetra: Ifpack_DynamicFactory::RegisterPreconditioner("ShyLU", buildShyLU);
    //     auto prec = Ifpack2::Factory::create<MAT>("SCHWARTZ", A);
    //     params.set("schwarz: overlap level", overlap);
    //     prec.setParameters(params);

    //     // if (verbose) {
    //     //     std::cout << "--- Ifpack_ShyLU was registered with"
    //     //     <<" Ifpack_DynamicFactory" << std::endl;
    //     // }
    // }

    // Read the XML parameter file. The resulting Teuchos parameter list
    // contains sublists for the partitioner, the preconditioner, linear solver etc.
    RCP<ParameterList> globalParams = rcp(new ParameterList());
    globalParams = Teuchos::getParametersFromXmlFile(parameterFileName);

    // Load global parameters
    std::string matrixFileName = globalParams->get<std::string>("matrix file name", "wathenSmall.mtx");
    std::string rhsFileName = globalParams->get<std::string>("rhs_file", "");
    std::string mapFileName = globalParams->get<std::string>("map_file", "");
    std::string precType = globalParams->get<std::string>("preconditioner type", "RILUK");
    int maxFiles = globalParams->get<int>("Maximum number of files to read in", 1);
    int startFile = globalParams->get<int>("Number of initial file", 1);
    int file_number = startFile;

    char filename[200];
    if (mapFileName != "" && maxFiles > 1) {
      mapFileName += "%d.mm";
      sprintf(filename, mapFileName.c_str(), file_number);
    } else {
      strcpy(filename, mapFileName.c_str());
    }
    if (verbose) {
      std::cout << "--- Using map file: " << filename << std::endl;
    }

    bool mapAvail = false;
    RCP<const Map> vecMap = NULL; 

    if (mapFileName != "") {
      mapAvail = true;
      auto M = Tpetra::MatrixMarket::Reader<BlockCrsMatrix>::readSparseFile(filename, comm);
      vecMap = M->getRowMap();
    }

    if (maxFiles > 1) {
      matrixFileName += "%d.mm";
      sprintf(filename, matrixFileName.c_str(), file_number);
    } else {
      strcpy(filename, matrixFileName.c_str());
    }

    if (verbose) {
      std::cout << "--- Using matrix file: " << filename << std::endl;
    }

    RCP<const CrsMatrix> A;
    if (mapAvail) {
      A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename, comm);
      map = A->getRowMap();
    } else {
      A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename, comm);
    }
    
    // if (err) {
    //   if (verbose) {
    //     std::cout << "!!! Matrix file could not be read in, info = " << err << std::endl;
    //   }
    //   std::cout << fail << std::endl;
    //   return -3;
    // }

    if (rhsFileName != "" && maxFiles > 1) {
      rhsFileName += "%d.mm";
      sprintf(filename, rhsFileName.c_str(), file_number);
    } else {
      strcpy(filename, rhsFileName.c_str());
    }
    if (verbose) {
      std::cout << "--- Using rhs file: " << filename << std::endl;
    }

    // Partition the matrix (old Isoroppia version)
    // ParameterList isorropiaParams = globalParams->sublist("Isorropia parameters");
    // TD: TODO: no Tpetra partitionner found in  Isorropia
    // RCP<Isorropia::Epetra::Partitioner> partitioner =
    //      rcp(new Isorropia::Epetra::Partitioner(A, isorropiaParams, false), true);
    // partitioner->partition();
    // RCP<Isorropia::Epetra::Redistributor> rd = rcp(new Isorropia::Epetra::Redistributor(partitioner));

    // TD: Consider now that if Zoltan2 enabled to use Zoltan2 preconditioner as 2 preconditioners might be available
  #if defined(HAVE_SHYLU_DDCORE_ZOLTAN2)
    partitioner = rcp(new Ifpack2::Zoltan2Partitioner<RowGraph>(A_->getGraph()));
  #else
    partitioner = rcp (new Ifpack2::LinearPartitioner<RowGraph>(A_->getGraph()));
  #end
    partitioner->computePartitions(); // TD: not sure




    // TD: can't find Tpetra alternative
    // const tpetra_import_type import(src_map, tgt_map);
    // rd = Tpetra::Distributor &distributor = import.getDistributor();

    int numRows = A->getGlobalNumRows();
    RCP<Map> vectorMap = rcp(new Map(numRows, 0, comm));
    MultiVector *RHS;
    bool isRHS = false;
    
    if (rhsFileName != "") {
      if (mapAvail) {
        Tpetra::MatrixMarket::Reader<MultiVector>::readVectorFile(rhsFileName, comm, vecMap);
      } else {
        Tpetra::MatrixMarket::Reader<MultiVector>::readVectorFile(rhsFileName, comm, vectorMap);
      }
    } else {
      // Generate a RHS and LHS
      isRHS = true;
      if (mapAvail) {
        RHS = new MultiVector(vecMap, 1);
      } else {
        RHS = new MultiVector(vectorMap, 1);
      }
      RHS->randomize();
    }

    MultiVector *LHS;
    if (mapAvail) {
      LHS = new MultiVector(vecMap, 1);
    } else {
      LHS = new MultiVector(*vectorMap, 1);
    }
    LHS->putScalar(0.0);
    // In Belos:
    // RCP<Vector> vecLHS = rcp(new V(map));
    // RCP<Vector> vecRHS = rcp(new V(map));
    // RCP<MultiVector> LHS, RHS;
    // .. stuff
    // LHS = Teuchos::rcp_implicit_cast<MV>(vecLHS);
    // RHS = Teuchos::rcp_implicit_cast<MV>(vecRHS);

    // Redistribute matrix and vectors
    // // OLD
    // RCP<CrsMatrix> rcpA;
    // RCP<MultiVector> rcpRHS;
    // RCP<MultiVector> rcpLHS;
    // RCP<CrsMatrix> newA;
    // MultiVector *newLHS, *newRHS;
    RCP<CrsMatrix> rcpA;
    RCP<MultiVector> vecRHS;
    RCP<MultiVector> vecLHS;
    RCP<CrsMatrix> newA;
    MultiVector *LHS, *RHS;

    if (mapAvail) {
      vecRHS = rcp(RHS, true);
      rcpLHS = rcp(LHS, true);
    } else {
      // rd->redistribute(*A, newA);
      delete A;
      rcpA = rcp(newA, true);

      rd->redistribute(*RHS, newRHS);
      delete RHS;
      vecRHS = rcp(newRHS, true);

      rd->redistribute(*LHS, newLHS);
      delete LHS;
      rcpLHS = rcp(newLHS, true);
    }

    /*if (!globalParams->isSublist("ML parameters")) {
      if (verbose) {
        std::cout << "!!! ML parameter list not found. Exiting." << std::endl;
      }
      std::cout << fail << std::endl;
      return -4;
    }
    ParameterList mlParameters = globalParams->sublist("ML parameters");*/
    // TD: because ML Epetra preconditioner will be replaced by a RILUK preconditioner
    // we need to replace ML parameters by the new preconditioner parameters
    if (!globalParams->isSublist(sprintf("%s parameters", precType))) { // TD: we have to add it to parameter file
      if (verbose) {
        std::cout << "!!! " << precType << " parameter list not found. Exiting." << std::endl;
      }
      std::cout << fail << std::endl;
      return -4;
    }

    if (!globalParams->isSublist("Belos parameters")) {
      if (verbose) {
        std::cout << "!!! Belos parameter list not found. Exiting." << std::endl;
      }
      std::cout << fail << std::endl;
      return -5;
    }
    ParameterList belosParams = globalParams->sublist("Belos parameters");
    int belosBlockSize = belosParams.get<int>("Block Size", 1);
    int belosMaxRestarts = belosParams.get<int>("Maximum Restarts", 0);
    int belosMaxIterations = belosParams.get<int>("Maximum Iterations", numRows);
    double belosTolerance = belosParams.get<double>("Convergence Tolerance", 1e-10);
    if (verbose) {
      std::cout << std::endl;
      std::cout << "--- Dimension of matrix: " << numRows << std::endl;
      std::cout << "--- Block size used by solver: " << belosBlockSize << std::endl;
      std::cout << "--- Number of restarts allowed: " << belosMaxRestarts << std::endl;
      std::cout << "--- Max number of Gmres iterations per restart cycle: " << belosMaxIterations << std::endl;
      std::cout << "--- Relative residual tolerance: " << belosTolerance << std::endl;
    }

    RCP<const CrsMatrix> iterA;
    RCP<const CrsMatrix> redistA;
    RCP<const MultiVector> iterb1;
    
    // RCP<ML_Epetra::MultiLevelPreconditioner> MLprec;
    RCP<Preconditioner> myPrec;

    Teuchos::Time setupPrecTimer("preconditioner setup timer", false);
    Teuchos::Time linearSolverTimer("linear solver timer");

    RCP<Belos::SolverManager<ST, MV, OP> > solver;
    while (file_number < maxFiles + startFile) {
      /*if (file_number == startFile)
      {*/
      // Build preconditioner (Epetra version used ML)
      if (mapAvail) {
        setupPrecTimer.start();
        // Epetra: MLprec = rcp(new ML_Epetra::MultiLevelPreconditioner(*A, mlParameters, false), true);
        // Epetra: MLprec->ComputePreconditioner();
        myPrec = Ifpack2::Factory::create<MAT>(precType, A);
        aPrec->compute();
        setupPrecTimer.stop();
      } else {
        setupPrecTimer.start();
        // Epetra: MLprec = rcp(new ML_Epetra::MultiLevelPreconditioner(*newA, mlParameters, false), true);
        // Epetra: MLprec->ComputePreconditioner();
        myPrec = Ifpack2::Factory::create<MAT>(precType, newA);
        setupPrecTimer.stop();
      }
      /*}
      else
      {
          setupPrecTimer.start();
          MLprec->ReComputePreconditioner();
          setupPrecTimer.stop();
      }*/

      // Build linear solver

      // Epetra: RCP<Belos::EpetraPrecOp> belosPrec = rcp(new Belos::EpetraPrecOp(MLprec), false);

      // Construct a preconditioned linear problem
      RCP<Belos::LinearProblem<ST, MV, OP> > problem =
          rcp(new Belos::LinearProblem<ST, MV, OP>(rcpA, rcpLHS, rcpRHS));
      problem->setRightPrec(prec);

      if (!problem->setProblem()) {
        if (verbose) {
          std::cout << "!!! Belos::LinearProblem failed to set up correctly."
                    << " Exiting." << std::endl;
        }
        std::cout << fail << std::endl;
        return -6;
      }

      // Create an iterative solver manager
      solver = rcp(new Belos::BlockGmresSolMgr<ST, MV, OP>(problem, rcp(&belosParams, false)));
      // Solve linear system
      linearSolverTimer.start();
      Belos::ReturnType ret = solver->solve();
      linearSolverTimer.stop();
      if (ret == Belos::Unconverged) {
        if (verbose) {
          std::cout << "!!! Linear solver did not converge to prescribed"
                    << " precision. Test failed." << std::endl;
        }
        std::cout << fail << std::endl;
        return -7;
      }
      // Print time measurements
      int numIters = solver->getNumIters();
      double timeSetupPrec = setupPrecTimer.totalElapsedTime();
      double timeLinearSolver = linearSolverTimer.totalElapsedTime();
      if (verbose) {
        std::cout << "--- Preconditioner setup time: " << timeSetupPrec << std::endl;
        std::cout << "--- Number of iterations performed: " << numIters << std::endl;
        std::cout << "--- Time to GMRES convergence: " << timeLinearSolver << std::endl;
        std::cout << "--- Average time per GMRES iteration: " << timeLinearSolver / numIters << std::endl;
        std::cout << "--- Total time to solution (prec + GMRES): " << timeSetupPrec + timeLinearSolver << std::endl;
      }

      file_number++;
      if (file_number >= maxFiles + startFile) {
        if (redistA != NULL) {
          delete redistA;
          redistA = NULL;
        }
        if (iterb1 != NULL) {
          delete iterb1;
          iterb1 = NULL;
        }
        break;
      } else {
        sprintf(filename, matrixFileName.c_str(), file_number);

        if (redistA != NULL) {
          delete redistA;
          redistA = NULL;
        }
        // Load the new matrix
        // if (mapAvail) {
        //   err = EpetraExt::MatrixMarketFileToCrsMatrix(filename, *vecMap, redistA);
        // } else {
        //   err = EpetraExt::MatrixMarketFileToCrsMatrix(filename, comm, iterA);
        // }
        if (mapAvail) {
          redistA = Tpetra::MatrixMarket::Reader<MultiVector>::readSparseFile(rhsFileName, comm);
          vecMap = redistA->getRowMap();
        } else {
          iterA = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename, comm);
        }
        if (err != 0) {
          if (myPID == 0)
            std::cout << "Could not open file: " << filename << std::endl;
          success = false;
        } else {
          if (mapAvail) {
            InitMatValues(*redistA, A);
            if (redistA != NULL) {
              delete redistA;
              redistA = NULL;
            }
          } else {
            // Tpetra alternative ?
            // rd->redistribute(*iterA, redistA);
            if (iterA != NULL) {
              delete iterA;
              iterA = NULL;
            }
            initMatValues(*redistA, newA);
          }
        }

        // Load the new rhs
        if (!allOneRHS) {
          sprintf(filename, rhsFileName.c_str(), file_number);

          if (iterb1 != NULL) {
            delete iterb1;
            iterb1 = NULL;
          }
          if (mapAvail) {
            err = EpetraExt::MatrixMarketFileToMultiVector(filename, *vecMap, iterb1);
          } else {
            err = EpetraExt::MatrixMarketFileToMultiVector(filename, *vectorMap, RHS);
          }
          if (err != 0) {
            if (myPID == 0)
              std::cout << "Could not open file: " << filename << std::endl;
            success = false;
          } else {
            if (mapAvail) {
              initMVValues(*iterb1, RHS);
              if (iterb1 != NULL) {
                delete iterb1;
                iterb1 = NULL;
              }
            } else {
              rd->redistribute(*RHS, iterb1);
              if (RHS != NULL) {
                delete RHS;
                RHS = NULL;
              }
              initMVValues(*iterb1, newRHS);
              // Should we delete iterb1
            }
          }
        }
      }
    }

    if (success)
      std::cout << "End Result: TEST PASSED" << std::endl;
    else
      std::cout << "End Result: TEST FAILED" << std::endl;

    myPrec.reset();
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return 0;
}

int main(int argc, char **argv) {
  return run<double>(argc, argv);
  // return run<float>(argc, argv);
}