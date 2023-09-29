
SET(HEADERS "")
SET(SOURCES "")

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  Basis.hpp
  FiniteElementProblem.hpp
#  Problem_Interface.hpp
#  Tcubed_FiniteElementProblem.hpp
  Pitchfork_FiniteElementProblem.hpp # --> WIP, TESTING
#  NormConstraint.hpp
#  LOCALinearConstraint.hpp --> WIP
)

APPEND_SET(SOURCES
  Basis.cpp
#  Problem_Interface.cpp
#  Tcubed_FiniteElementProblem.cpp
  Pitchfork_FiniteElementProblem.cpp # --> WIP, TESTING
#  NormConstraint.cpp
#  LOCALinearConstraint.cpp --> WIP
)

# Add this below:
# IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_EPETRA AND NOX_ENABLE_LOCA) 

IF(NOX_ENABLE_LOCA)

  TRIBITS_ADD_LIBRARY(
    locatpetratestproblems
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    TESTONLY
    DEPLIBS loca locatpetra
  )

ENDIF()
