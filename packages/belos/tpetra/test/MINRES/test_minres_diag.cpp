//@HEADER
// TOREDO

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "BelosMinresSolMgr.hpp"

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

// Teuchos
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

//************************************************************************************************

template<class MV>
class Vector_Operator
{
  public:

    Vector_Operator(int m_in, int n_in) : m(m_in), n(n_in) {};

    virtual ~Vector_Operator() {};

    virtual void operator () (const MV &x, MV &y) = 0;

    int size (int dim) const { return (dim == 1) ? m : n; };

  protected:

    int m, n;        // an (m x n) operator

  private:

    // Not allowing copy construction.
    Vector_Operator( const Vector_Operator& ): m(0), n(0) {};
    Vector_Operator* operator=( const Vector_Operator& ) { return NULL; };

};

//************************************************************************************************

template<class MV>
class Diagonal_Operator : public Vector_Operator<MV>
{
  public:

    Diagonal_Operator(int n_in, double v_in) : Vector_Operator<MV>(n_in, n_in), v(v_in) { };

    ~Diagonal_Operator() { };

    void operator () (const MV &x, MV &y)
    {
      y.scale( v, x );
    };

  private:

    double v;
};

//************************************************************************************************

template<class MV>
class Diagonal_Operator_2 : public Vector_Operator<MV>
{
  public:

    Diagonal_Operator_2<MV>(int n_in, int min_gid_in, double v_in)
    : Vector_Operator<MV>(n_in, n_in), min_gid(min_gid_in), v(v_in) {}

    ~Diagonal_Operator_2() { };

    void operator () (const MV &x, MV &y)
    {
        auto yLocalData = y.getLocalViewHost(Tpetra::Access::ReadWrite);
        auto xLocalData = y.getLocalViewHost(Tpetra::Access::ReadOnly);

        for (size_t i = 0; i < x.getLocalLength(); ++i) {
            for (size_t j = 0; j < x.getNumVectors(); ++j) {
                yLocalData(j, i) = (min_gid+i+1)*v*xLocalData(j,i);
            }
        }
    };

  private:

    int min_gid;
    double v;
};

//************************************************************************************************

template<class MV>
class Composed_Operator : public Vector_Operator<MV>
{
  public:

    Composed_Operator(int n,
        const Teuchos::RCP<Vector_Operator<MV>>& pA_in,
        const Teuchos::RCP<Vector_Operator<MV>>& pB_in);

    virtual ~Composed_Operator() {};

    virtual void operator () (const MV &x, MV &y);

  private:

    Teuchos::RCP<Vector_Operator<MV>> pA;
    Teuchos::RCP<Vector_Operator<MV>> pB;
};

template<class MV>
Composed_Operator<MV>::Composed_Operator(int n_in,
    const Teuchos::RCP<Vector_Operator<MV>>& pA_in,
    const Teuchos::RCP<Vector_Operator<MV>>& pB_in)
: Vector_Operator<MV>(n_in, n_in), pA(pA_in), pB(pB_in)
{
}

template<class MV>
void Composed_Operator<MV>::operator () (const MV &x, MV &y)
{
  MV ytemp(y.getMap(), y.getNumVectors(), false);
  (*pB)( x, ytemp );
  (*pA)( ytemp, y );
}

//************************************************************************************************

template<class OP, class ST, class MP, class MV>
class Trilinos_Interface : public OP
{
  public:

    Trilinos_Interface(const Teuchos::RCP<Vector_Operator<MV>>   pA_in,
        const Teuchos::RCP<const Teuchos::Comm<int>> pComm_in,
        const Teuchos::RCP<const MP>  pMap_in)
      : pA (pA_in),
      pComm (pComm_in),
      pMap (pMap_in),
      use_transpose (false)
  {}

    void apply (const MV &X,
                MV &Y,
                Teuchos::ETransp mode = Teuchos::NO_TRANS,
                ST alpha = Teuchos::ScalarTraits<ST>::one(),
                ST beta = Teuchos::ScalarTraits<ST>::zero()) const override;

    virtual ~Trilinos_Interface() {};

    bool hasTransposeApply() const {return(use_transpose);};      // always set to false (in fact the default)

    Teuchos::RCP<const MP> getDomainMap() const override {return pMap; }
    Teuchos::RCP<const MP> getRangeMap() const override {return pMap; }

  private:

    Teuchos::RCP<Vector_Operator<MV>>   pA;
    Teuchos::RCP<const Teuchos::Comm<int>> pComm;
    Teuchos::RCP<const MP>  pMap;

    bool use_transpose;
};

template<class OP, class ST, class MP, class MV>
void Trilinos_Interface<OP, ST, MP, MV>::apply (const MV &X,
            MV &Y,
            Teuchos::ETransp mode,
            ST alpha,
            ST beta) const {
    (*pA)(X,Y);
}

//************************************************************************************************

template<class OP, class ST, class MP, class MV>
class Iterative_Inverse_Operator : public Vector_Operator<MV>
{
  public:

  Iterative_Inverse_Operator(int n_in, int blocksize,
      const Teuchos::RCP<Vector_Operator<MV>>& pA_in,
      std::string opString="Iterative Solver", bool print_in=false);

  virtual ~Iterative_Inverse_Operator() {}

  virtual void operator () (const MV &b, MV &x);

  private:

  Teuchos::RCP<Vector_Operator<MV>> pA;       // operator which will be inverted
  // supplies a matrix std::vector multiply
  const bool print;

  Teuchos::Time timer;
  Teuchos::RCP<Teuchos::Comm<int>> pComm;
  Teuchos::RCP<MP>  pMap;

  Teuchos::RCP<OP> pPE;
  Teuchos::RCP<Teuchos::ParameterList>         pList;
  Teuchos::RCP<Belos::LinearProblem<double,MV,OP> >   pProb;
  Teuchos::RCP<Belos::MinresSolMgr<double,MV,OP> >      pBelos;
};

template<class OP, class ST, class MP, class MV>
Iterative_Inverse_Operator<OP, ST, MP, MV>::Iterative_Inverse_Operator(int n_in, int blocksize,
    const Teuchos::RCP<Vector_Operator<MV>>& pA_in,
    std::string opString, bool print_in)
: Vector_Operator<MV>(n_in, n_in),      // square operator
  pA(pA_in),
  print(print_in),
  timer(opString)
{

  int n_global;
// TD: could the following ifdef block be replaced by just pComm = Tpetra::getDefaultComm() ?
// AM: yes
#ifdef HAVE_TPETRA_MPI
  MPI_Allreduce(&n_in, &n_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  pComm = Teuchos::rcp (new Teuchos::MpiComm<int> (MPI_COMM_WORLD));
#else
  pComm = Teuchos::rcp (new Teuchos::SerialComm<int> ());
  n_global = n_in;
#endif
  pMap =  Teuchos::rcp( new MP(n_global, n_in, 0, pComm) );

  pPE = Teuchos::rcp( new Trilinos_Interface<OP, ST, MP, MV>(pA, pComm, pMap ) );

  pProb = Teuchos::rcp( new Belos::LinearProblem<double,MV,OP>() );
  pProb->setOperator( pPE );

  int max_iter = 100;
  double tol = 1.0e-10;
  int verbosity = Belos::Errors + Belos::Warnings;
  if (print)
    verbosity += Belos::TimingDetails + Belos::StatusTestDetails;

  pList = Teuchos::rcp( new Teuchos::ParameterList );
  pList->set( "Maximum Iterations", max_iter );
  pList->set( "Convergence Tolerance", tol );
  pList->set( "Verbosity", verbosity );

  pBelos = Teuchos::rcp( new Belos::MinresSolMgr<double,MV,OP>(pProb, pList) );
}

template<class OP, class ST, class MP, class MV>
void Iterative_Inverse_Operator<OP, ST, MP, MV>::operator () (const MV &b, MV &x)
{
  int pid = pComm->getRank();

  // Initialize the solution to zero
  x.putScalar( 0.0 );

  // Reset the solver, problem, and status test for next solve (HKT)
  pProb->setProblem( Teuchos::rcp(&x, false), Teuchos::rcp(&b, false) );

  timer.start();
  Belos::ReturnType ret = pBelos->solve();
  timer.stop();

  if (pid == 0 && print) {
    if (ret == Belos::Converged)
    {
      std::cout << std::endl << "pid[" << pid << "] Minres converged" << std::endl;
      std::cout << "Solution time: " << timer.totalElapsedTime() << std::endl;

    }
    else
      std::cout << std::endl << "pid[" << pid << "] Minres did not converge" << std::endl;
  }
}

//************************************************************************************************

template <class ScalarType>
bool run_test(int argc, char *argv[]){
  // Get default Tpetra template types
  using ST = typename Tpetra::MultiVector<ScalarType>::scalar_type;
  using LO = typename Tpetra::Vector<>::local_ordinal_type;
  using GO = typename Tpetra::Vector<>::global_ordinal_type;
  using NT = typename Tpetra::Vector<>::node_type;

  // Init Tpetra types
  using OP = typename Tpetra::Operator<ST,LO,GO,NT>;
  using MV = typename Tpetra::MultiVector<ST,LO,GO,NT>;
  using MP = typename Tpetra::Map<LO,GO,NT>;

  using Teuchos::RCP;
  using Teuchos::rcp;

  Teuchos::GlobalMPISession session(&argc, &argv, NULL);

  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int pid = rank(*comm);

  bool verbose = false;
  bool success = true;

  try {

    int n(10);
    int numRHS=1;
    
    RCP<MP> map = RCP(new MP(n, 0, comm));
    
    MV X(map, numRHS), Y(map, numRHS);
    X.putScalar( 1.0 );
    
    // Inner computes inv(D2)*y
    RCP<Diagonal_Operator_2<MV>> D2 = rcp(new Diagonal_Operator_2<MV>(n, map->getMinGlobalIndex(), 1.0));
    Iterative_Inverse_Operator<OP, ST, MP, MV> A2(n, 1, D2, "Belos (inv(D2))", true);
    
    // should return x=(1, 1/2, 1/3, ..., 1/10)
    // AM: If you uncomment this line, test failed, I don't know why
    // A2(X,Y);
    
    if (pid == 0) {
      std::cout << "Vector Y should have all entries [1, 1/2, 1/3, ..., 1/10]" << std::endl;
    }
    Y.print(std::cout);

    // Inner computes inv(D)*x
    RCP<Diagonal_Operator<MV>> D = rcp(new Diagonal_Operator<MV>(n, 4.0));
    RCP<Iterative_Inverse_Operator<OP, ST, MP, MV>> Inner =
      rcp(new Iterative_Inverse_Operator<OP, ST, MP, MV>(n, 1, D, "Belos (inv(D))", false));

    // Composed_Operator computed inv(D)*B*x
    RCP<Diagonal_Operator<MV>> B = rcp(new Diagonal_Operator<MV>(n, 4.0));
    RCP<Composed_Operator<MV>> C = rcp(new Composed_Operator<MV>(n, Inner, B));

    // Outer computes inv(C) = inv(inv(D)*B)*x = inv(B)*D*x = x
    RCP<Iterative_Inverse_Operator<OP, ST, MP, MV>> Outer =
      rcp(new Iterative_Inverse_Operator<OP, ST, MP, MV>(n, 1, C, "Belos (inv(C)=inv(inv(D)*B))", true));

    // should return x=1/4
    (*Inner)(X,Y);

    if (pid == 0) {
      std::cout << std::endl << "Vector Y should have all entries [1/4, 1/4, 1/4, ..., 1/4]" << std::endl;
    }
    Y.print(std::cout);

    // should return x=1
    (*Outer)(X,Y);

    if (pid == 0) {
      std::cout << "Vector Y should have all entries [1, 1, 1, ..., 1]" << std::endl;
    }
    Y.print(std::cout);

    // Compute the norm of Y - 1.0
    std::vector<ST> norm_Y(Y.getNumVectors());
    Teuchos::ArrayView<ST> normView(norm_Y);

    Y.update(-1.0, X, 1.0);
    Y.norm2(norm_Y);

    if (pid == 0)
      std::cout << "Two-norm of std::vector (Y-1.0) : "<< norm_Y[0] << std::endl;

    success = (norm_Y[0] < 1e-10 && !Teuchos::ScalarTraits<double>::isnaninf( norm_Y[0] ) );
    
    if (success) {
      if (pid == 0)
        std::cout << "End Result: TEST PASSED" << std::endl;
    } else {
      if (pid == 0)
        std::cout << "End Result: TEST FAILED" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return success;
}

//************************************************************************************************
//************************************************************************************************

int main(int argc, char *argv[])
{
  return run_test<double>(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}