
#ifndef PIKE_GLOBAL_COMM_HPP
#define PIKE_GLOBAL_COMM_HPP

#include <mpi.h>
#include <mutex>

namespace pike {

static std::mutex mpi_mutex;
static MPI_Comm Global_MPI_Comm = MPI_COMM_WORLD;

inline void initialize_global_comm(MPI_Comm comm) {
    std::lock_guard<std::mutex> guard(mpi_mutex);
    Global_MPI_Comm = comm;
}

inline MPI_Comm get_global_comm() {
    std::lock_guard<std::mutex> guard(mpi_mutex);
    return Global_MPI_Comm;
}

}

#endif /* PIKE_GLOBAL_COMM_HPP */