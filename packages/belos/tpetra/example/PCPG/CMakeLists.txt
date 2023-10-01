
TRIBITS_ADD_EXECUTABLE(
  PCPG_Tpetra_File_Ex
  SOURCES PCPGTpetraExFile.cpp
  COMM serial mpi
  # ARGS --verbose
  )
