//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

// Purpose
// The example tests the successive right-hand sides capabilities of ML
// and Belos on a heat flow u_t = u_xx problem.
//
// A sequence of linear systems with the same coefficient matrix and
// different right-hand sides is solved.  A seed space is generated dynamically,
// and a deflated linear system is solved.  After each solves, the first
// few Krylov vectors are saved, and used to reduce the number of iterations
// for later solves.
// The optimal numbers of vectors to deflate and save are not known.
// Presently, the maximum number of vectors to deflate (seed space dimension)
// and to save are user paraemters.
// The seed space dimension is less than or equal to total number of vectors saved.
// The difference between the seed space dimension and the total number of vectors,
// is the number of vectors used to update the seed space after each solve.
// I guess that a seed space whose dimension is a small fraction of the total space
// will be best.
//
// maxSave=1 and maxDeflate=0 uses no recycling (not tested ).
//
// TODO: Instrument with timers, so that we can tell what is going on besides
//       by counting the numbers of iterations.

// Adapted from test_pcpg_epetraex.cpp by David M. Day (with original comments)



// #ifdef TPETRA_MPI // CWS FIND REPLACEMENT
// // #include <Epetra_MpiComm.h<
// #else
// // #include <Epetra_SerialComm.h<
// #endif

// Tpetra
#include <Tpetra_Map_fwd.hpp>
#include <Tpetra_Vector_fwd.hpp>
#include <Tpetra_CrsMatrix_fwd.hpp>

// MueLu
#include <MueLu_CreateTpetraPreconditioner.hpp> // includes MueLu.hpp

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosPCPGSolMgr.hpp>
#include <BelosMueLuAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTpetraOperator.hpp>

// Teuchos
#include <Teuchos_RCP.hpp> // included in MueLu.hpp
#include <Teuchos_Comm.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp> // included in MueLu.hpp
#include <Teuchos_ParameterList.hpp> // included in MueLu.hpp
#include <Teuchos_CommandLineProcessor.hpp> // included in MueLu.hpp
#include <Teuchos_StandardCatchMacros.hpp>


// template<class ScalarType>
int main(int argc, char *argv[]) {
    // using SC = typename Tpetra::Vector<ScalarType>::scalar_type;
    using SC = typename Tpetra::Vector<double>::scalar_type;
    using LO = typename Tpetra::Vector<>::local_ordinal_type;
    using GO = typename Tpetra::Vector<>::global_ordinal_type;
    using NT = typename Tpetra::Vector<>::node_type;

    using SCT = typename Teuchos::ScalarTraits<SC>;
    using MT  = typename SCT::magnitudeType;
    using MV  = typename Tpetra::MultiVector<SC,LO,GO,NT>;
    using OP  = typename Tpetra::Operator<SC,LO,GO,NT>;
    using MVT = typename Belos::MultiVecTraits<SC,MV>;
    using OPT = typename Belos::OperatorTraits<SC,MV,OP>;

    using tcrsmatrix_t   = Tpetra::CrsMatrix<SC,LO,GO,NT>;
    using tmap_t         = Tpetra::Map<LO,GO,NT>;
    using tvector_t      = Tpetra::Vector<SC,LO,GO,NT>;
    using tmultivector_t = Tpetra::MultiVector<SC,LO,GO,NT>;

    using toperator_t  = Tpetra::Operator<SC,LO,GO,NT>;
    using mtoperator_t = MueLu::TpetraOperator<SC,LO,GO,NT>;
    // using btprecop_t = // find something;

    using scarray_t = Teuchos::ArrayView<SC>;
    using goarray_t = Teuchos::ArrayView<GO>;

    using Teuchos::ParameterList; // all of this may look fine but do not be fooled ...
    using Teuchos::RCP;           // it is not so clear what any of this does
    using Teuchos::rcp;
    using Teuchos::Comm;
    using Teuchos::rcp_dynamic_cast;

    int MyPID = 0;
    int numProc = 1;
    // CWS: TODO initialize MPI and get communicator
    const auto comm = Tpetra::getDefaultComm();

    // Laplace's equation, homogenous Dirichlet boundary counditions, [0,1]^2
    // regular mesh, Q1 finite elements
    bool success = false;
    bool verbose = false;

    try {
        bool proc_verbose = false;
        int frequency = -1;        // frequency of status test output.
        int blocksize = 1;         // blocksize, PCPGIter
        int numrhs = 1;            // number of right-hand sides to solve for
        int maxiters = 30;         // maximum number of iterations allowed per linear system

        int maxDeflate = 4; // maximum number of vectors deflated from the linear system;
        // There is no overhead cost assoc with changing maxDeflate between solves
        int maxSave = 8;    // maximum number of vectors saved from current and previous .");
        // If maxSave changes between solves, then re-initialize (setSize).

        // Hypothesis: seed vectors are conjugate.
        // Initial versions allowed users to supply a seed space et cetera, but no longer.

        // The documentation it suitable for certain tasks, like defining a modules grammar,
        std::string ortho("ICGS"); // The Belos documentation obscures the fact that
        // IMGS is Iterated Modified Gram Schmidt,
        // ICGS is Iterated Classical Gram Schmidt, and
        // DKGS is another Iterated Classical Gram Schmidt.
        // Mathematical issues, such as the difference between ICGS and DKGS, are not documented at all.
        // UH tells me that Anasazi::SVQBOrthoManager is available;  I need it for Belos
        MT tol = 1.0e-8;           // relative residual tolerance

        // How do command line parsers work?
        Teuchos::CommandLineProcessor cmdp(false,true);

        cmdp.setOption("verbose","quiet",&verbose,"Print messages and results");
        cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters)");
        cmdp.setOption("tol",&tol,"Relative residual tolerance used by PCPG solver");
        cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for");
        cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size)");
        cmdp.setOption("num-deflate",&maxDeflate,"Number of vectors deflated from the linear system");
        cmdp.setOption("num-save",&maxSave,"Number of vectors saved from old Krylov subspaces");
        cmdp.setOption("ortho-type",&ortho,"Orthogonalization type, either DGKS, ICGS or IMGS");

        if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
            return -1;
        }
        if (!verbose)
            frequency = -1;  // reset frequency if test is not verbose

        ////////////////////////////////////////////////////
        //                Form the problem                //
        ////////////////////////////////////////////////////

        int numElePerDirection = 14 * numProc; // CWS: why 14?
        int num_time_step = 4; // CWS: why 4?
        int numNodes = (numElePerDirection - 1)*(numElePerDirection - 1);

        //By the way, either matrix has (3*numElePerDirection - 2)^2 nonzeros.
        RCP<tmap_t> Map         = rcp(new tmap_t(numNodes, 0, comm));
        RCP<tcrsmatrix_t> Stiff = rcp(new tcrsmatrix_t(Map, 0));
        RCP<tcrsmatrix_t> Mass  = rcp(new tcrsmatrix_t(Map, 0));
        RCP<tvector_t> vecLHS   = rcp(new tvector_t(Map));
        RCP<tvector_t> vecRHS   = rcp(new tvector_t(Map));
        RCP<tmultivector_t> LHS, RHS;

        SC ko = 8.0 / 3.0, k1 = -1.0 / 3.0;
        scarray_t ko_arr(&ko, 1);
        scarray_t k1_arr(&k1, 1);

        SC h = 1.0 / static_cast<SC>(numElePerDirection);  // x=(iX,iY)h

        SC mo = h*h*4.0/9.0, m1 = h*h/9.0, m2 = h*h/36.0;
        scarray_t mo_arr(&mo, 1);
        scarray_t m1_arr(&m1, 1);
        scarray_t m2_arr(&m2, 1);

        SC pi = 4.0*atan(1.0), valueLHS;
        GO lid, node, iX, iY, pos;

        goarray_t pos_arr(&pos, 1);

        for (lid = Map->getMinLocalIndex(); lid <= Map->getMaxLocalIndex(); lid++) {

            node = Map->getGlobalElement(lid);
            iX  = node  % (numElePerDirection-1);
            iY  = ( node - iX )/(numElePerDirection-1);
            Stiff->insertGlobalValues(node, pos_arr, ko_arr); // global row ID, global col ID, value
            Mass->insertGlobalValues(node, pos_arr, mo_arr); // init guess violates hom Dir bc
            valueLHS = sin( pi*h*((SC) iX+1) )*cos( 2.0 * pi*h*((SC) iY+1) );
            vecLHS->replaceGlobalValue(node, valueLHS);

            if (iY > 0) {
                pos_arr[0] = iX + (iY-1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); //North
                Mass->insertGlobalValues(node, pos_arr, m1_arr);
            }

            if (iY < numElePerDirection-2) {
                pos_arr[0] = iX + (iY+1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); //South
                Mass->insertGlobalValues(node, pos_arr, m1_arr);
            }

            if (iX > 0) {
                pos_arr[0] = iX-1 + iY*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // West
                Mass->insertGlobalValues(node, pos_arr, m1_arr);
                if (iY > 0) {
                pos_arr[0] = iX-1 + (iY-1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // North West
                Mass->insertGlobalValues(node, pos_arr, m2_arr);
                }
                if (iY < numElePerDirection-2) {
                pos_arr[0] = iX-1 + (iY+1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // South West
                Mass->insertGlobalValues(node, pos_arr, m2_arr);
                }
            }

            if (iX < numElePerDirection - 2) {
                pos_arr[0] = iX+1 + iY*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // East
                Mass->insertGlobalValues(node, pos_arr, m1_arr);
                if (iY > 0) {
                pos_arr[0] = iX+1 + (iY-1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // North East
                Mass->insertGlobalValues(node, pos_arr, m2_arr);
                }
                if (iY < numElePerDirection-2) {
                pos_arr[0] = iX+1 + (iY+1)*(numElePerDirection-1);
                Stiff->insertGlobalValues(node, pos_arr, k1_arr); // South East
                Mass->insertGlobalValues(node, pos_arr, m2_arr);
                }
            }
        }
        Stiff->fillComplete();
        if (!Stiff->isStorageOptimized()) {
            // do something
            // for now:
            std::cout << "Stiff Matrix storage is not optimized" << std::endl;
        }
        Mass->fillComplete();
        if (!Mass->isStorageOptimized()) {
            // do something
            std::cout << "Mass Matrix storage is not optimized" << std::endl;
        }

        SC one = 1.0, hdt = .00005; // half time step

        const RCP<tcrsmatrix_t> A = rcp(new tcrsmatrix_t(*Stiff) ); // A = Mass+Stiff*dt/2
        try {
            Tpetra::MatrixMatrix::Add(*Mass, false, one, *A, hdt);
        } catch (std::runtime_error& ex) {
            std::cout << "Error from MatrixMatrix::Add: " << ex.what() << std::endl;
            return 1;
        }


        A->fillComplete();
        if (!A->isStorageOptimized()) {
            std::cout << "A storage is not optimized" << std::endl;
        }

        hdt = -hdt;
        RCP<tcrsmatrix_t> B = rcp(new tcrsmatrix_t(*Stiff) ); // B = Mass-Stiff*dt/2
        try {
            Tpetra::MatrixMatrix::Add(*Mass, false, one, *B,hdt);
        } catch (std::runtime_error& ex) {
            std::cout << "Error from MatrixMatrix::Add: " << ex.what() << std::endl;
            return 1;
        }

        B->fillComplete();
        if (!B->isStorageOptimized()) {
            std::cout << "B storage is not optimized" << std::endl;
        }
        B->apply(*vecLHS, *vecRHS); // rhs_new := B*lhs_old,

        proc_verbose = verbose && (MyPID==0);  /* Only print on the zero processor */

        LHS = Teuchos::rcp_implicit_cast<tmultivector_t>(vecLHS);
        RHS = Teuchos::rcp_implicit_cast<tmultivector_t>(vecRHS);

        ////////////////////////////////////////////////////
        //            Construct Preconditioner            //
        ////////////////////////////////////////////////////

        ParameterList MueLuList; // Set MueLuList for Smoothed Aggregation

        MueLuList.set("smoother: type","Chebyshev"); // Chebyshev smoother  ... aztec??
        MueLuList.set("smoother: sweeps",3);
        MueLuList.set("smoother: pre or post", "both"); // both pre- and post-smoothing
#ifdef HAVE_MUELU_AMESOS2
        std::cout << "HAVE_MUELU_AMESOS2 active" << std::endl; // CWS: remove when done
        MueLuList.set("coarse: type", "Amesos2-KLU2"); // solve with serial direct solver KLU (CWS: CHECK)
#else
        MueLuList.set("coarse: type", "Jacobi"); // not recommended
        puts("Warning: Iterative coarse grid solve");
#endif

        const RCP<toperator_t> A_operator = rcp_dynamic_cast<toperator_t>(A);
        RCP<mtoperator_t> Prec = MueLu::CreateTpetraPreconditioner(A_operator, MueLuList);

        assert(Prec != Teuchos::null);

        // Create the Belos preconditioned operator from the preconditioner.
        // NOTE:  This is necessary because Belos expects an operator to apply the
        //        preconditioner with Apply() NOT ApplyInverse().
        // RCP<btprecop_t> belosPrec = rcp(new btprecop_t(Prec));

        ///////////////////////////////////////////////////
        //             Create Parameter List             //
        ///////////////////////////////////////////////////

        const size_t NumGlobalElements = RHS->getGlobalLength();
        if (maxiters == -1)
        maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run

        ParameterList belosList;
        belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
        belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
        belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
        belosList.set( "Num Deflated Blocks", maxDeflate );    // Number of vectors in seed space
        belosList.set( "Num Saved Blocks", maxSave );          // Number of vectors saved from old spaces
        belosList.set( "Orthogonalization", ortho );           // Orthogonalization type

        if (numrhs > 1) {
        belosList.set( "Show Maximum Residual Norm Only", true );  // although numrhs = 1.
        }
        if (verbose) {
        belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
            Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails );
        if (frequency > 0)
            belosList.set( "Output Frequency", frequency );
        }
        else
        belosList.set( "Verbosity", Belos::Errors + Belos::Warnings + Belos::FinalSummary );

        ///////////////////////////////////////////////////
        //    Construct Preconditioned Linear Problem    //
        ///////////////////////////////////////////////////

        RCP<Belos::LinearProblem<SC,MV,OP> > problem
            = rcp( new Belos::LinearProblem<SC,MV,OP>( A, LHS, RHS ) );
        problem->setLeftPrec( Prec );

        bool set = problem->setProblem();
        if (set == false) {
            if (proc_verbose) {
                std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
            }
            return -1;
        }

        // Create an iterative solver manager.
        RCP< Belos::SolverManager<SC,MV,OP> > solver
        = rcp( new Belos::PCPGSolMgr<SC,MV,OP>(problem, rcp(&belosList,false)) );

        ////////////////////////////////////////////////////
        //                  Iterate PCPG                  //
        ////////////////////////////////////////////////////

        if (proc_verbose) {
            std::cout << std::endl << std::endl;
            std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
            std::cout << "Number of right-hand sides: " << numrhs << std::endl;
            std::cout << "Block size used by solver: " << blocksize << std::endl;
            std::cout << "Maximum number of iterations allowed: " << maxiters << std::endl;
            std::cout << "Relative residual tolerance: " << tol << std::endl;
            std::cout << std::endl;
        }
        bool badRes;
        for( int time_step = 0; time_step < num_time_step; time_step++){
        if (time_step) {
            B->apply(*LHS, *RHS); // rhs_new := B*lhs_old,
            set = problem->setProblem(LHS, RHS);
            if (set == false) {
                if (proc_verbose)
                    std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
                return -1;
            }
        } // if time_step
        std::vector<SC> rhs_norm(numrhs);
        MVT::MvNorm(*RHS, rhs_norm);
        std::cout << "\t\t\t\tRHS norm is ... " << rhs_norm[0] << std::endl;

        // Perform solve

        Belos::ReturnType ret = solver->solve();

        // Compute actual residuals.

        badRes = false;
        std::vector<SC> actual_resids(numrhs);
        tmultivector_t resid(Map, numrhs);
        OPT::Apply( *A, *LHS, resid );
        MVT::MvAddMv( -1.0, resid, 1.0, *RHS, resid );
        MVT::MvNorm( resid, actual_resids );
        MVT::MvNorm( *RHS, rhs_norm );
        std::cout << "\t\t\t\tRHS norm is ... " << rhs_norm[0] << std::endl;

        if (proc_verbose) {
            std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
            for ( int i=0; i<numrhs; i++) {
                SC actRes = actual_resids[i]/rhs_norm[i];
                std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
                if (actRes > tol) badRes = true;
            }
        }

        success = ret==Belos::Converged && !badRes;
        if (!success)
            break;
        } // for time_step

        if (success) {
            if (proc_verbose)
                std::cout << "End Result: TEST PASSED" << std::endl;
        } else {
            if (proc_verbose)
                std::cout << "End Result: TEST FAILED" << std::endl;
        }

    } // try block
    TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    // MPI_Finalize();

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
} // main