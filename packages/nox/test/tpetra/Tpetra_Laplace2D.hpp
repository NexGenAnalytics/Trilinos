#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_Map_fwd.hpp"
#include "Tpetra_MultiVector.hpp" // fwd doesn't have default enabled template types
#include "Tpetra_Vector_fwd.hpp"
#include "Tpetra_Export_fwd.hpp"
#include "Tpetra_Import_fwd.hpp"
#include "Tpetra_RowGraph_fwd.hpp"
#include "Tpetra_RowMatrix_fwd.hpp"
#include "Tpetra_CrsGraph_fwd.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_Operator_fwd.hpp"

// Typedefs and constants

typedef Tpetra::MultiVector<>::scalar_type Scalar;
typedef Tpetra::MultiVector<>::local_ordinal_type LO;
typedef Tpetra::MultiVector<>::global_ordinal_type GO;
typedef Tpetra::MultiVector<>::node_type Node;

typedef Tpetra::Map<LO, GO, Node> Map;
typedef Tpetra::Vector<Scalar, LO, GO, Node> TV;
typedef Tpetra::CrsMatrix<Scalar, LO, GO, Node> CRSM;
typedef Tpetra::RowMatrix<Scalar, LO, GO, Node> TRM;
typedef Tpetra::Operator<Scalar, LO, GO, Node> TO;
typedef Tpetra::CrsGraph< LO, GO, Node> CSRG;

// Interfaces

namespace NOX
{
  namespace Tpetra
  {
    namespace Interface
    {

      /*! \brief Used by NOX::Tpetra to provide a link to the
        external code for Jacobian fills.

        This is only required if the user wishes to supply their own Jacobian
        operator.
      */
      class Jacobian
      {

      public:
        //! Constructor.
        Jacobian(){};

        //! Destructor.
        virtual ~Jacobian(){};

        /*! Compute Jacobian given the specified input vector x.  Returns
          true if computation was successful.
         */
        virtual bool computeJacobian(const TV &x, TO &Jac) = 0;
      };

      /*! \brief Used by NOX::Tpetra to provide a link to the
        external code for Precondtioner fills.

        This is only required if the user wishes to supply their own
        preconditioner operator.
      */
      class Preconditioner
      {

      public:
        //! Constructor
        Preconditioner(){};

        //! Destructor
        virtual ~Preconditioner(){};

        //! Computes a user defined preconditioner.
        virtual bool computePreconditioner(const TV &x,
                                           TO &M,
                                           Teuchos::ParameterList *precParams = 0) = 0;
      };

      /*!
        \brief Supplies NOX with the set nonlinear equations.

        This is the minimum required information to solve a nonlinear
        problem using the NOX::Tpetra objects for the linear algebra
        implementation.  Used by NOX::Tpetra::Group to provide a link
        to the external code for residual fills.
      */
      class Required
      {

      public:
        //! Type of fill that a computeF() method is used for.
        /*! computeF() can be called for a variety of reasons:

        - To evaluate the function residuals.
        - To be used in an approximation to the Jacobian (finite difference or directional derivative).
        - To be used in an approximation to the preconditioner.

        This flag tells computeF() what the evaluation is used for.  This allows the user to change the fill process to eliminate costly terms.  For example, sometimes, terms in the function are very expensive and can be ignored in a Jacobian calculation.  The user can query this flag and determine not to recompute such terms if the computeF() is used in a Jacobian calculation.
         */
        enum FillType
        {
          //! The exact residual (F) is being calculated.
          Residual,
          //! The Jacobian matrix is being estimated.
          Jac,
          //! The preconditioner matrix is being estimated.
          Prec,
          //! The fill context is from a FD approximation (includes FDC)
          FD_Res,
          //! The fill context is from a MF approximation
          MF_Res,
          //! The fill context is from a MF computeJacobian() approximation
          MF_Jac,
          //! A user defined estimation is being performed.
          User
        };

        //! Constructor
        Required(){};

        //! Destructor
        virtual ~Required(){};

        //! Compute the function, F, given the specified input vector x.  Returns true if computation was successful.
        virtual bool computeF(const TV &x, TV &F,
                              const FillType fillFlag) = 0;
      };
    } // namespace Interface
  }   // namespace Tpetra
} // namespace NOX

namespace Laplace2D
{
  // this is required to know the number of lower, upper, left and right
  // node for each node of the Cartesian grid (composed by nx \timex ny
  // elements)

  void getMyNeighbours(const int i, const int nx, const int ny,
                  int &left, int &right,
                  int &lower, int &upper);

  // This function creates a CrsMatrix, whose elements corresponds
  // to the discretization of a Laplacian over a Cartesian grid,
  // with nx grid point along the x-axis and and ny grid points
  // along the y-axis. For the sake of simplicity, I suppose that
  // all the nodes in the matrix are internal nodes (Dirichlet
  // boundary nodes are supposed to have been already condensated)

  CRSM * createLaplacian(const int nx, const int ny, const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

  // ==========================================================================
  // This class contians the main definition of the nonlinear problem at
  // hand. A method is provided to compute F(x) for a given x, and another
  // method to update the entries of the Jacobian matrix, for a given x.
  // As the Jacobian matrix J can be written as
  //    J = L + diag(lambda*exp(x[i])),
  // where L corresponds to the discretization of a Laplacian, and diag
  // is a diagonal matrix with lambda*exp(x[i]). Basically, to update
  // the jacobian we simply update the diagonal entries. Similarly, to compute
  // F(x), we reset J to be equal to L, then we multiply it by the
  // (distributed) vector x, then we add the diagonal contribution
  // ==========================================================================

} // namespace Laplace2D

class PDEProblem
{

public:
  // constructor. Requires the number of nodes along the x-axis
  // and y-axis, the value of lambda, and the communicator
  // (to define a Map, which is a linear map in this case)
  PDEProblem(const int nx, const int ny, const double lambda,
             const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

  // destructor
  ~PDEProblem();

  // compute F(x)
  void computeF(const TV &x, TV &f);

  // update the Jacobian matrix for a given x
  void updateJacobian(const TV &x);

  // returns a pointer to the internally stored matrix
  CRSM *getMatrix()
  {
    return matrix_;
  }

private:
  int nx_, ny_;
  double hx_, hy_;
  CRSM *matrix_;
  double lambda_;

}; /* class PDEProblem */

// ==========================================================================
// This is the main NOX class for this example. Here we define
// the interface between the nonlinear problem at hand, and NOX.
// The constructor accepts a PDEProblem object. Using a pointer
// to this object, we can update the Jacobian and compute F(x),
// using the definition of our problem. This interface is bit
// crude: For instance, no PrecMatrix nor Preconditioner is specified.
// ==========================================================================

class SimpleProblemInterface : public NOX::Tpetra::Interface::Required,
                               public NOX::Tpetra::Interface::Jacobian,
                               public NOX::Tpetra::Interface::Preconditioner
{

public:
  //! Constructor
  SimpleProblemInterface(PDEProblem *problem) : problem_(problem){};

  //! Destructor
  ~SimpleProblemInterface(){};

  bool computeF(const TV &x, TV &f,
                NOX::Tpetra::Interface::Required::FillType F)
  {
    problem_->computeF(x, f);
    return true;
  };

  bool computeJacobian(const TV &x, TV &Jac)
  {
    problem_->updateJacobian(x);
    return true;
  }

  bool computePreconditioner(const TV &x, TO &Op, Teuchos::ParameterList *)
  {
    problem_->updateJacobian(x);
    return true;
  }

  bool computePrecMatrix(const TV &x, TRM &M)
  {
    std::cout << "*ERR* SimpleProblem::preconditionVector()\n";
    std::cout << "*ERR* don't use explicit preconditioning" << std::endl;
    throw 1;
  }

private:
  PDEProblem *problem_;
}; /* class SimpleProblemInterface */
