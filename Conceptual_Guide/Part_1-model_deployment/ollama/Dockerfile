# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /models
# Install dependencies
RUN apt update && apt install -y curl unzip

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Expose Ollama API port
EXPOSE 11434

# COPY model
COPY llama-2-7b.Q8_0.gguf /models/llama-2-7b.Q8_0.gguf

# Create model from gguf file
RUN ollama serve & sleep 10 \
    && ollama create ollama-llama-2-7b-q8

# Copy Ollama run script
COPY ollama_run.sh /models/ollama_run.sh

# Make the script executable
RUN chmod +x /models/ollama_run.sh

# Run Ollama and load the model
ENTRYPOINT [ "/models/ollama_run.sh" ]