
SET(HEADERS "")
SET(SOURCES "")

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  Basis.hpp
#  FiniteElementProblem.hpp
#  Problem_Interface.hpp
#  Tcubed_FiniteElementProblem.hpp
#  Pitchfork_FiniteElementProblem.hpp
#  NormConstraint.hpp
#  LOCALinearConstraint.hpp
)

APPEND_SET(SOURCES
#  Problem_Interface.cpp
#  Tcubed_FiniteElementProblem.cpp
#  Pitchfork_FiniteElementProblem.cpp
#  NormConstraint.cpp
#  LOCALinearConstraint.cpp
)

IF(#[[Change this `NOX_ENABLE_ABSTRACT_IMPLEMENTATION_EPETRA` for Tpetra]] #[[AND]] NOX_ENABLE_LOCA)

  TRIBITS_ADD_LIBRARY(
    locatpetratestproblems
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    TESTONLY
    DEPLIBS loca locatpetra
  )

ENDIF()
