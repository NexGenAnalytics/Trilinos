# Choose a base image
FROM ubuntu:22.04

# Install prerequisites
RUN apt-get update && \
    apt-get install -y build-essential git curl python3 gfortran libssl-dev

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/ && cd /opt/ && git clone --depth 1 --branch "v0.20.1" https://github.com/spack/spack.git

COPY . /opt/src/Trilinos

RUN . /opt/spack/share/spack/setup-env.sh && spack compiler find
RUN . /opt/spack/share/spack/setup-env.sh && spack external find --not-buildable && spack external list

## Make trilinos env
RUN mkdir -p /opt/spack-trilinos-env
ADD ./spack-trilinos-depends.yaml /opt/spack-trilinos-env/spack-trilinos-depends.yaml
RUN mv /opt/spack-trilinos-env/spack-trilinos-depends.yaml /opt/spack-trilinos-env/spack.yaml

# create pre_trilinos environment from spack.yaml and concretize
RUN cd /opt/spack-trilinos-env \
  && . /opt/spack/share/spack/setup-env.sh && spack env create pre_trilinos /opt/spack-trilinos-env/spack.yaml\
  && spack env activate pre_trilinos && spack concretize && spack env deactivate

# make trilinos env from lock
RUN . /opt/spack/share/spack/setup-env.sh && spack env create trilinos /opt/spack/var/spack/environments/pre_trilinos/spack.lock

# activate trilinos env and install
RUN . /opt/spack/share/spack/setup-env.sh && spack env activate trilinos && spack install --fail-fast && spack gc -y && spack env deactivate

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/build/Trilinos

# Build using the spack environment we created
RUN bash /opt/src/Trilinos/nga-ci/build.sh

RUN bash /opt/src/Trilinos/nga-ci/test.sh
