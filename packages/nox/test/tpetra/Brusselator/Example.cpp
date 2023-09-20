
// AM: REDO THE HEADER!!!

#include <Tpetra_Core.hpp>

#include <Teuchos_StandardCatchMacros.hpp>

// User's application specific files
#include "Problem_Interface.hpp"
#include "Brusselator.hpp"

template <typename ScalarType>
int run(int argc, char *argv[])
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  
  using tvector_t = typename Tpetra::Vector<ST, LO, GO>;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &std::cout);
  RCP< const Teuchos::Comm<int> > Comm = Tpetra::getDefaultComm();
  const int MyPID = Comm->getRank();
  const int NumProc = Comm->getSize();

  bool verbose = true;
  bool success = false;

  try {
    // AM: Before Epetra timer, redo?

    // Check for verbose option
    if (argc > 1)
      if (argv[1][0]=='-' && argv[1][1]=='v')
        verbose = true;
    
    // Set the problem size (1000 elements)
    int NumGlobalNodes = 1001;

    // The number of unknowns must be at least equal to the
    // number of processors.
    if (NumGlobalNodes < NumProc) {
      std::cout << "numGlobalNodes = " << NumGlobalNodes
                << " cannot be < number of processors = " << NumProc << std::endl;
      std::cout << "Test failed!" << std::endl;
      exit(1);
    }

    // Create the Brusselator problem class. This creates all required
    // Tpetra objects for the problem and allows calls to the
    // function (F) and Jacobian evaluation routines.
    //Brusselator<int>::OverlapType OType = Brusselator<int>::OverlapType::NODES;
    OverlapType OType = NODES;
    RCP<Brusselator<ST>> Problem = rcp(new Brusselator<ST>(NumGlobalNodes, Comm, OType));

    // Get the vector from the Problem
    RCP<tvector_t> soln = Problem->getSolution();
    // NOX::Epetra::Vector noxSoln(soln, NOX::Epetra::Vector::CreateView);    // AM: TOFIX

    // Begin Nonlinear Solver ************************************

    // Create the top level parameter list
    RCP<ParameterList> nlParamsPtr = rcp(new ParameterList);
    Teuchos::ParameterList& nlParams = *(nlParamsPtr.get());


    // AM: TODO below vvvvvv
    // AM: TODO below vvvvvv



  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}

int main(int argc, char *argv[]) {
  return run<double>(argc, argv);
  // return run<float>(argc, argv);
}
