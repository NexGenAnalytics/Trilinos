# Choose a base image
FROM calebschilly/trilinos-deps:main

COPY . /opt/src/Trilinos

RUN mkdir -p /opt/build/Trilinos

RUN useradd -ms /bin/bash nga-ci

# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build.sh

RUN chown nga-ci /opt

USER nga-ci

RUN bash /opt/src/Trilinos/nga-ci/test.sh
