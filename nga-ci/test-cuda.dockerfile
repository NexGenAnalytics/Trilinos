FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git

# Fix paths
RUN export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Retrieve CUDA samples
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/cuda-samples.git

# build sample
WORKDIR /opt/cuda-samples/Samples/1_Utilities/deviceQuery
RUN make dbg=1 TARGET_ARCH=x86_64 HOST_COMPILER=g++ CUDA_PATH=/usr/local/cuda-12.2

# Run sample
WORKDIR /opt/cuda-samples/bin/x86_64/linux/debug
RUN ./deviceQuery
