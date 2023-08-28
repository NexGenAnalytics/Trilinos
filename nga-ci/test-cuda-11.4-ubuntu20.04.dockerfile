FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
RUN apt-get update && apt-get install -y git

# Fix paths
RUN export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# Retrieve CUDA samples
WORKDIR /opt
RUN git clone --branch v11.4.1 https://github.com/NVIDIA/cuda-samples.git

# build sample
WORKDIR /opt/cuda-samples/Samples/deviceQuery
RUN make dbg=1 TARGET_ARCH=x86_64 HOST_COMPILER=g++ CUDA_PATH=/usr/local/cuda-11.4

# Run sample
WORKDIR /opt/cuda-samples/bin/x86_64/linux/debug
RUN ./deviceQuery
