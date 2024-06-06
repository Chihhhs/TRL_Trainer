#!/bin/bash
curl -fsSL https://ollama.com/install.sh | sh

wget https://huggingface.co/Xcvddax/llama2-chat-attck-gguf/resolve/main/llama-2-attack-Q4_K_M.gguf?download=true -O llama-2-attack-Q4_K_M.gguf

ollama create -f Modelfile chih/llama-2-chat-attack
ollama push chih/llama-2-chat-attack

echo "Set up Ollama successfully!"