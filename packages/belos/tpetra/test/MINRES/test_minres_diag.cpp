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
//
// This test generates diagonal matrices for MINRES to solve.
//
// NOTE: No preconditioner is used in this case.
//

// Belos includes
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "BelosMinresSolMgr.hpp"
// Tpetra includes
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.h"
#include "Tpetra_MultiVector.hpp"
// Teuchos includes
#include "Teuchos_Comm.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

//
/* TD: add typedefs as it is the common way in tpetra tests */
// Get default Tpetra template types
typedef Tpetra::MultiVector<>::scalar_type ST;
typedef Tpetra::Vector<>::local_ordinal_type LO;
typedef Tpetra::Vector<>::global_ordinal_type GO;
typedef Tpetra::Vector<>::node_type NT;
// Init Tpetra types
typedef Tpetra::Operator<ST,LO,GO,NT> OP;
typedef Tpetra::MultiVector<ST,LO,GO,NT> MV;
typedef Tpetra::Map<LO,GO,NT> MP;

//************************************************************************************************

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

class Diagonal_Operator : public Vector_Operator
{
  public:

    Diagonal_Operator(int n_in, double v_in) : Vector_Operator(n_in, n_in), v(v_in) { };

    ~Diagonal_Operator() { };

    void operator () (const MV &x, MV &y)
    {
      y.scale( v, x );
    };

  private:

    double v;
};

//************************************************************************************************

class Diagonal_Operator_2 : public Vector_Operator
{
  public:

    Diagonal_Operator_2(int n_in, int min_gid_in, double v_in)
    : Vector_Operator(n_in, n_in), min_gid(min_gid_in), v(v_in) {}

    ~Diagonal_Operator_2() { };

    void operator () (const MV &x, MV &y)
    {
      int myCols = y.MyLength();
      for (int j=0; j < x.getNumVectors(); ++j) {
        for (int i=0; i < myCols; ++i) (*y(j))[i] = (min_gid+i+1)*v*(*x(j))[i];  // NOTE: square operator!
      }
    };

  private:

    int min_gid;
    double v;
};

//************************************************************************************************

class Composed_Operator : public Vector_Operator
{
  public:

    Composed_Operator(int n,
        const Teuchos::RCP<Vector_Operator>& pA_in,
        const Teuchos::RCP<Vector_Operator>& pB_in);

    virtual ~Composed_Operator() {};

    virtual void operator () (const MV &x, MV &y);

  private:

    Teuchos::RCP<Vector_Operator> pA;
    Teuchos::RCP<Vector_Operator> pB;
};

Composed_Operator::Composed_Operator(int n_in,
    const Teuchos::RCP<Vector_Operator>& pA_in,
    const Teuchos::RCP<Vector_Operator>& pB_in)
: Vector_Operator(n_in, n_in), pA(pA_in), pB(pB_in)
{
}

void Composed_Operator::operator () (const MV &x, MV &y)
{
  MV ytemp(y.Map(), y.getNumVectors(), false);
  (*pB)( x, ytemp );
  (*pA)( ytemp, y );
}

//************************************************************************************************

class Trilinos_Interface : public OP
{
  public:

    Trilinos_Interface(const RCP<Vector_Operator>   pA_in,
        const Teuchos::RCP<const Teuchos::Comm<int>> pComm_in,
        const Teuchos::RCP<const MP>  pMap_in)
      : pA (pA_in),
      pComm (pComm_in),
      pMap (pMap_in),
      use_transpose (false)
  {}

    int apply(const MV& X, MV& Y) const;

    /* TD: epetra only operator method
    int applyInverse(const MV& X, MV& Y) const
    {
      return(apply(X,Y));  // No inverse
    };*/

    virtual ~Trilinos_Interface() {};

    // const char * Label() const {return("Trilinos_Interface, an Operator implementation");}; only in epetra ?

   /* TD: Epetra version used virtual bool UseTranspose() */
    bool hasTransposeApply() const {return(use_transpose);};      // always set to false (in fact the default)

    /* TD: Only Epetra Operator has a setter
    int SetUseTranspose(bool UseTranspose_in) { use_transpose = false; return(-1); };
    */

    /* TD: HasNormInf and NormInf not in TPetra_Operator.hpp. only EPetra
    bool HasNormInf() const {return(false);};                // cannot return inf-norm
    double NormInf() const {return(0.0);};
    */

    /* TD: Not in TPetra_Operator.hpp. only EPetra 
    virtual const Teuchos::RCP<const Teuchos::Comm<int>> & Comm() const {return *pComm; }
    */

    /* TD: Epetra version was: virtual const MP & OperatorDomainMap() const {return *pMap; } */
    virtual Teuchos::RCP<const MP> > getDomainMap() const {return *pMap; }

    /* TD: Epetra version was virtual const MP & OperatorRangeMap() const {return *pMap; }  */
    virtual Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > getRangeMap() const {return *pMap; }

  private:

    Teuchos::RCP<Vector_Operator>   pA;
    Teuchos::RCP<const Teuchos::Comm> pComm;
    Teuchos::RCP<const MP>  pMap;

    bool use_transpose;
};

int Trilinos_Interface::Apply(const MV& X, MV& Y) const
{
  (*pA)(X,Y);

  return(0);
}

//************************************************************************************************

class Iterative_Inverse_Operator : public Vector_Operator
{
  public:

  Iterative_Inverse_Operator(int n_in, int blocksize,
      const Teuchos::RCP<Vector_Operator>& pA_in,
      std::string opString="Iterative Solver", bool print_in=false);

  virtual ~Iterative_Inverse_Operator() {}

  virtual void operator () (const MV &b, MV &x);

  private:

  Teuchos::RCP<Vector_Operator> pA;       // operator which will be inverted
  // supplies a matrix std::vector multiply
  const bool print;

  Teuchos::Time timer;
  Teuchos::RCP<Teuchos::Comm> pComm;
  Teuchos::RCP<MP>  pMap;

  Teuchos::RCP<OP> pPE;
  Teuchos::RCP<Teuchos::ParameterList>         pList;
  Teuchos::RCP<LinearProblem<double,MV,OP> >   pProb;
  Teuchos::RCP<MinresSolMgr<double,MV,OP> >      pBelos;
};

Iterative_Inverse_Operator::Iterative_Inverse_Operator(int n_in, int blocksize,
    const RCP<Vector_Operator>& pA_in,
    std::string opString, bool print_in)
: Vector_Operator(n_in, n_in),      // square operator
  pA(pA_in),
  print(print_in),
  timer(opString)
{

  int n_global;
// TD: could the following ifdef block be replaced by just pComm = Tpetra::getDefaultComm() ?
#ifdef HAVE_TPETRA_MPI
  MPI_Allreduce(&n, &n_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  pComm = Teuchos::rcp (new Teuchos::MpiComm<int> (MPI_COMM_WORLD));
#else
  pComm = Teuchos::rcp (new Teuchos::SerialComm<int> ());
  n_global = n;
#endif
  pMap =  Teuchos::rcp( new MP(n_global, n, 0, *pComm) );

  pPE = Teuchos::rcp( new Trilinos_Interface(pA, pComm, pMap ) );

  pProb = Teuchos::rcp( new LinearProblem<double,MV,OP>() );
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

  pBelos = Teuchos::rcp( new MinresSolMgr<double,MV,OP>(pProb, pList) );
}

void Iterative_Inverse_Operator::operator () (const MV &b, MV &x)
{
  int pid = pComm->MyPID();

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
//************************************************************************************************

int main(int argc, char *argv[])
{
  int pid = -1;

  Teuchos::GlobalMPISession session(&argc, &argv, NULL);

  RCP<const Comm<int> > comm = Tpetra::getDefaultComm();

  bool verbose = false;
  bool success = true;
  try {

    pid = Teuchos::rank(*comm);

    int n(10);
    int numRHS=1;

    MP map = MP(n, 0, Comm);

    MV X(map, numRHS), Y(map, numRHS);
    X.putScalar( 1.0 );

    // Inner computes inv(D2)*y
    Teuchos::RCP<Diagonal_Operator_2> D2 = Teuchos::rcp(new Diagonal_Operator_2(n, map.MinMyGID(), 1.0));
    Iterative_Inverse_Operator A2(n, 1, D2, "Belos (inv(D2))", true);

    // should return x=(1, 1/2, 1/3, ..., 1/10)
    A2(X,Y);

    if (pid==0) {
      std::cout << "Vector Y should have all entries [1, 1/2, 1/3, ..., 1/10]" << std::endl;
    }
    Y.Print(std::cout);

    // Inner computes inv(D)*x
    Teuchos::RCP<Diagonal_Operator> D = Teuchos::rcp(new Diagonal_Operator(n, 4.0));
    Teuchos::RCP<Iterative_Inverse_Operator> Inner =
      Teuchos::rcp(new Iterative_Inverse_Operator(n, 1, D, "Belos (inv(D))", false));

    // Composed_Operator computed inv(D)*B*x
    Teuchos::RCP<Diagonal_Operator> B = Teuchos::rcp(new Diagonal_Operator(n, 4.0));
    Teuchos::RCP<Composed_Operator> C = Teuchos::rcp(new Composed_Operator(n, Inner, B));

    // Outer computes inv(C) = inv(inv(D)*B)*x = inv(B)*D*x = x
    Teuchos::RCP<Iterative_Inverse_Operator> Outer =
      Teuchos::rcp(new Iterative_Inverse_Operator(n, 1, C, "Belos (inv(C)=inv(inv(D)*B))", true));

    // should return x=1/4
    (*Inner)(X,Y);

    if (pid==0) {
      std::cout << std::endl << "Vector Y should have all entries [1/4, 1/4, 1/4, ..., 1/4]" << std::endl;
    }
    Y.Print(std::cout);

    // should return x=1
    (*Outer)(X,Y);

    if (pid==0) {
      std::cout << "Vector Y should have all entries [1, 1, 1, ..., 1]" << std::endl;
    }
    Y.Print(std::cout);

    // Compute the norm of Y - 1.0
    std::vector<double> norm_Y(Y.getNumVectors());
    Y.update(-1.0, X, 1.0);
    Y.norm2(&norm_Y[0]);

    if (pid==0)
      std::cout << "Two-norm of std::vector (Y-1.0) : "<< norm_Y[0] << std::endl;

    success = (norm_Y[0] < 1e-10 && !Teuchos::ScalarTraits<double>::isnaninf( norm_Y[0] ) );

    if (success) {
      if (pid==0)
        std::cout << "End Result: TEST PASSED" << std::endl;
    } else {
      if (pid==0)
        std::cout << "End Result: TEST FAILED" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
