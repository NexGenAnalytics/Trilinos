TRIBITS_ADD_EXECUTABLE_AND_TEST(
    test_parser
    SOURCES test_parser.cpp
    COMM serial mpi
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(ParserTestFiles
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_FILES config.yaml
  EXEDEPS test_parser
  )
