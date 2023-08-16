# Choose a base image
FROM calebschilly/trilinos-deps:main

ARG USER=nga-ci
ARG UID=1000
USER nga-ci

COPY . /opt/src/Trilinos
RUN mkdir -p /opt/build/Trilinos

# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build.sh

RUN bash /opt/src/Trilinos/nga-ci/test.sh
