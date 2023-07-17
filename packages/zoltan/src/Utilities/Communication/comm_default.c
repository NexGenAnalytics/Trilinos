#include "comm.h"

#ifdef __cplusplus
/* if C++, define the rest of this header file as extern C */
extern "C" {
#endif

/* Function to get the default communicator */
MPI_Comm MPI_Comm_Default() {
  return MPI_COMM_WORLD;
}

#ifdef __cplusplus
} /* closing bracket for extern "C" */
#endif