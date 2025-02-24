# Use the NVIDIA Triton Server with TensorRT LLM support and CUDA 12.6
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# Set the working directory
WORKDIR /app

# Copy the local Llama-3.2-1B model to the container
COPY meta-llama-3.2-1B.tar /app/model/meta-llama-3.2-1B.tar

# Extract the Llama-3.2-1B model
RUN mkdir -p /app/model && \
    tar -xvf /app/model/meta-llama-3.2-1B.tar -C /app/model/

# Git clone https://github.com/NVIDIA/TensorRT-LLM.git.
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git

RUN apt update -y \
    && apt install -y --no-install-recommends cmake git git git-lfs \
    && cd /app/TensorRT-LLM \
    && git submodule update --init --recursive \
    && apt update -y \
    && apt install -y tensorrt-dev \
    && git lfs install \
    && git lfs pull

# Copy scripts
COPY convert_model.sh /app/
COPY prepare_configs.sh /app/
COPY exec_triton_server.sh /app/
COPY cmake.sh /app/

# Make scripts executable
RUN chmod +x /app/convert_model.sh \
    && chmod +x /app/prepare_configs.sh \
    && chmod +x /app/exec_triton_server.sh \
    && chmod +x /app/cmake.sh

# Convert model, prepare model repository and run the Triton Inference Server.
ENTRYPOINT ["/bin/bash", "/app/exec_triton_server.sh"]
