#!/bin/bash

# Enable debugging
set -e  # Exit on any error
set -x  # Print each command before executing

# Convert the model
echo "Running model conversion..."
./convert_model.sh

# Prepare the model repository
echo "Preparing model repository..."
./prepare_configs.sh

# Make the kernels
# echo "Building custom kernels..."
# cd /app/cmake.sh

# Run the Triton Inference Server
echo "Starting Triton Inference Server..."
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/

# Keep the container alive
echo "Keeping container alive for..."
tail -f /dev/null
