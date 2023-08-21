# Choose a base image
FROM calebschilly/trilinos-deps-GPU:main AS build-stage

COPY . /opt/src/Trilinos
RUN mkdir -p /opt/build/Trilinos

# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build-gpu-epetraOFF.sh

FROM build-stage AS test-stage

ARG OMPI_ALLOW_RUN_AS_ROOT=1
ARG OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN bash /opt/src/Trilinos/nga-ci/test.sh

FROM scratch AS export-stage
COPY --from=test-stage /tmp/artifacts /tmp/artifacts
