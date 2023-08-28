FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git

# Get information using Driver Utility
RUN sudo apt install nvidia-utils-535
RUN nvidia-smi

# Retrieve CUDA samples
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/cuda-samples.git

# build sample
WORKDIR /opt/cuda-samples/Samples/1_Utilities/deviceQuery
RUN make dbg=1 TARGET_ARCH=x86_64 HOST_COMPILER=g++ CUDA_PATH=/usr/local/cuda-12.2

# Run sample
WORKDIR /opt/cuda-samples/bin/x86_64/linux/debug
RUN ./deviceQuery
