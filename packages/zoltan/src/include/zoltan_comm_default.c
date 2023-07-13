#include <mpi.h>

#ifdef __cplusplus
/* if C++, define the rest of this header file as extern C */
extern "C" {
#endif

/* Function to set the default communicator */
MPI_Comm Zoltan_Get_Default_Communicator() {
  return MPI_COMM_WORLD;
}

#ifdef __cplusplus
} /* closing bracket for extern "C" */
#endif