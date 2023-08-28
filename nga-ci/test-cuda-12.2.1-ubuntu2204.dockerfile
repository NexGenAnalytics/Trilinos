FROM ubuntu:22.04

# Install requirements
RUN apt-get update && apt-get install -y git wget libxml2

# Install Cuda 12.2.1
# from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04
WORKDIR /opt
RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
RUN sh cuda_12.2.1_535.86.10_linux.run

# Fix paths
RUN export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Retrieve CUDA samples
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/cuda-samples.git

# build sample
WORKDIR /opt/cuda-samples/Samples/1_Utilities/deviceQuery
RUN make dbg=1 TARGET_ARCH=x86_64 HOST_COMPILER=g++ CUDA_PATH=/usr/local/cuda-12.2

# Run sample
WORKDIR /opt/cuda-samples/bin/x86_64/linux/debug
RUN ./deviceQuery
