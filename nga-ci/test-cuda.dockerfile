FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git

# retrieve CUDA samples
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/cuda-samples.git
# build deviceQuery sample
WORKDIR /opt/cuda-samples/Samples/1_Utilities/deviceQuery
RUN make dbg=1 TARGET_ARCH=x86_64 HOST_COMPILER=g++