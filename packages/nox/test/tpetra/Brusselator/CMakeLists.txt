TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_THYRA)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tpetra_brusselator
    SOURCES
      Example.cpp
      Brusselator.hpp
      Basis.hpp
      Problem_Interface.hpp
    COMM
      serial mpi
    PASS_REGULAR_EXPRESSION
      "Test passed!"
  )

ENDIF()
