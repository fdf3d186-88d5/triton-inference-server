#!/bin/bash

export CUDA_ARCHITECTURES="75"
export HF_LLAMA_MODEL=/app/model/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/
export UNIFIED_CKPT_PATH=/tmp/ckpt/llama/1b/
export ENGINE_PATH=/tmp/engines/llama/1b/

python /app/examples/llama/convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
                 --remove_input_padding enable \
                 --gpt_attention_plugin auto \
                 --context_fmha disable \
                 --gemm_plugin float16 \
                 --output_dir ${ENGINE_PATH} \
                 --kv_cache_type paged \
                 --max_batch_size 1 \
                 --max_seq_len 512