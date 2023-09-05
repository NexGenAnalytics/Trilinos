# Choose a base image
FROM calebschilly/trilinos-deps:main AS build-stage

COPY . /opt/src/Trilinos
RUN mkdir -p /opt/build/Trilinos

# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build-mpi-epetraOFF.sh

# For running later
RUN chmod +x /opt/src/Trilinos/nga-ci/test-mpi.sh
