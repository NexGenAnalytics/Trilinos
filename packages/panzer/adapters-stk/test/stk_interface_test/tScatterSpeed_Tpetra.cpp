// STL includes
#include <iostream>
#include <vector>
#include <set>

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_DefaultComm.hpp>

// Intrepid2 includes
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"

// Panzer includes
#include "Panzer_ConnManager.hpp"
#include "Panzer_FieldPattern.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

// Panzer_STK includes
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STKConnManager.hpp"

// Tpetra includes
#include "Tpetra_Map.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"

using Teuchos::Array;
using Teuchos::ArrayRCP;
using Teuchos::ArrayView;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Time;
using Teuchos::TimeMonitor;

typedef double Scalar;
typedef Kokkos::DynRankView<Scalar, PHX::Device> FieldContainer;
typedef Tpetra::Map<panzer::LocalOrdinal, panzer::GlobalOrdinal> Map;
typedef Tpetra::CrsMatrix<Scalar, panzer::LocalOrdinal, panzer::GlobalOrdinal> CrsMatrix;
typedef Tpetra::CrsGraph<panzer::LocalOrdinal, panzer::GlobalOrdinal> CrsGraph;

//****************************Function Definitions******************************
void newAssembly(Teuchos::FancyOStream &out);
// Will run the generic setup of the DOFManager through the buildGlobalUnknowns
size_t setUp1(RCP<Map> &rowmap,
              RCP<Map> &colmap,
              RCP<panzer::DOFManager> &my_dofM,
              RCP<panzer::ConnManager> &conn);

// I'm not entirely sure on this yet.
void fillMeUp1(std::vector<std::vector<panzer::GlobalOrdinal>> &gids,
               std::vector<std::vector<panzer::LocalOrdinal>> &lids,
               std::vector<std::vector<std::vector<Scalar>>> &miniMat,
               RCP<panzer::DOFManager> &dofM,
               const std::vector<int> &mElem,
               const RCP<const Map> &mcmap);
RCP<Time> New_Time = TimeMonitor::getNewCounter("New Assembly Time");
RCP<Time> Old_Time = TimeMonitor::getNewCounter("Old Assembly Time");

int xelem = 10;
int yelem = 10;
int zelem = 10;
int xblocks = 1;
int yblocks = 1;
int zblocks = 1;

//******************************************************************************

int main(int argc, char *argv[])
{
  Teuchos::oblackholestream blackhole;
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
  // Teuchos::GlobalMPISession mpiSession(&argc,&argv, &std::cout);
#else
  EPIC_FAIL // Panzer is an MPI only code.
#endif
  Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
  out.setOutputToRootOnly(-1);
  out.setShowProcRank(true);
  newAssembly(out);

  TimeMonitor::summarize();

  return 0;
}

//******************************************************************************
//                          Standard Usage Functions
//******************************************************************************

void newAssembly(Teuchos::FancyOStream & /* out */)
{
  RCP<panzer::DOFManager> my_dofM;
  RCP<Map> rowmap;
  RCP<Map> colmap;
  RCP<panzer::ConnManager> conn;

  const std::vector<int> &myElements = conn->getElementBlock("eblock-0_0_0");

  std::vector<std::vector<panzer::GlobalOrdinal>> gids;
  std::vector<std::vector<panzer::LocalOrdinal>> lids;
  std::vector<std::vector<std::vector<Scalar>>> miniMat;
  fillMeUp1(gids, lids, miniMat, my_dofM, myElements, colmap);

  auto crsgraph = rcp(new CrsGraph(rowmap, colmap, 1));

  // Tell the graph where elements will be.
  for (size_t e = 0; e < myElements.size(); ++e)
  {
    for (size_t i = 0; i < gids[e].size(); ++i)
    {
      crsgraph->insertGlobalIndices(gids[e][i], gids[e].size(), &gids[e][0]);
    }
  }
  {
    crsgraph->fillComplete();
  }
  auto crsmat = rcp(new CrsMatrix(crsgraph));

  // Where the data transfer takes place.
  for (std::size_t i = 0; i < 20; i++)
  {
    Teuchos::TimeMonitor LocalTimer(*New_Time);

    for (size_t e = 0; e < myElements.size(); ++e)
    {
      for (size_t j = 0; j < gids[e].size(); ++j)
      {
        int accid = lids[e][j];
        crsmat->sumIntoGlobalValues(accid, lids[e].size(), &miniMat[e][j][0], &lids[e][0]);
      }
    }
  }

  return;
}

size_t setUp1(RCP<Map> &rowmap,
              RCP<Map> &colmap,
              RCP<panzer::DOFManager> &my_dofM,
              RCP<panzer::ConnManager> &conn)
{
  RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
  pl->set("X Blocks", xblocks);
  pl->set("Y Blocks", yblocks);
  pl->set("Z Blocks", zblocks);
  pl->set("X Elements", xelem);
  pl->set("Y Elements", yelem);
  pl->set("Z Elements", zelem);

  panzer_stk::CubeHexMeshFactory factory;
  factory.setParameterList(pl);
  RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);

  conn = rcp(new panzer_stk::STKConnManager(mesh));

  RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis1 = rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::exec_space, double, double>);
  RCP<const panzer::FieldPattern> pressure_pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis1));

  my_dofM = Teuchos::rcp(new panzer::DOFManager());

  my_dofM->setConnManager(conn, MPI_COMM_WORLD);

  my_dofM->addField("u", pressure_pattern);
  my_dofM->addField("v", pressure_pattern);
  my_dofM->addField("w", pressure_pattern);
  my_dofM->addField("p", pressure_pattern);

  my_dofM->buildGlobalUnknowns();

  std::vector<int> owned;
  std::vector<int> ownedAndGhosted;

  my_dofM->getOwnedIndicesAsInt(owned);
  my_dofM->getOwnedAndGhostedIndicesAsInt(ownedAndGhosted);

  size_t sz = ownedAndGhosted.size();

  Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

  rowmap = rcp(new Map(sz, ownedAndGhosted.data(), ownedAndGhosted.size(), 0, comm));
  colmap = rcp(new Map(sz, ownedAndGhosted.data(), ownedAndGhosted.size(), 0, comm));

  return sz;
}

void fillMeUp1(std::vector<std::vector<panzer::GlobalOrdinal>> &gids,
               std::vector<std::vector<panzer::LocalOrdinal>> &lids,
               std::vector<std::vector<std::vector<Scalar>>> &miniMat,
               RCP<panzer::DOFManager> &dofM,
               const std::vector<int> &mElem,
               const RCP<const Map> &mcmap)
{
  for (std::size_t e = 0; e < mElem.size(); ++e)
  {
    std::vector<panzer::GlobalOrdinal> tgids;
    dofM->getElementGIDsAsInt(mElem[e], tgids);
    std::vector<int> tlids;
    for (size_t i = 0; i < tgids.size(); ++i)
    {
      tlids.push_back(mcmap->getLocalElement(tgids[i]));
    }
    std::vector<std::vector<double>> tminiMat;
    for (size_t i = 0; i < tgids.size(); ++i)
    {
      std::vector<double> temp(tgids.size());
      for (size_t j = 0; j < tgids.size(); ++j)
      {
        // Right now everything is a one. That was just to make sure that the overlapping worked
        // correctly. This can literally be set to anything.
        double newval = 1;
        temp[j] = newval;
      }
      tminiMat.push_back(temp);
      temp.clear();
    }
    gids.push_back(tgids);
    lids.push_back(tlids);
    miniMat.push_back(tminiMat);
  }
}
