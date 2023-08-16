# Choose a base image
FROM calebschilly/trilinos-deps:main as pull_image

COPY . /opt/src/Trilinos
RUN mkdir -p /opt/build/Trilinos

FROM pull_image as build_stage
# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build.sh

FROM build_stage as test_stage
RUN bash /opt/src/Trilinos/nga-ci/test.sh
