#!/bin/bash

ollama serve &
sleep 10
ollama run ollama-llama-2-7b-q8 && tail -f /dev/null