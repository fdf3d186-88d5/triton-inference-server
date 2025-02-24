#!/bin/bash

curl -X POST http://localhost:11434/v1/completions -H "Content-Type: application/json" -d '{"model": "ollama-llama-2-7b-q8", "prompt": "Hello, how are you?"}'
