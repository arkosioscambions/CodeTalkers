#!/bin/bash
set -e 

# ===== Magicoder =====
echo "Setting up Magicoder..."
git clone https://github.com/ise-uiuc/magicoder.git

# Copy modified files
cp update/magicoder/llm_wrapper.py magicoder/src/magicoder/llm_wrapper.py
cp update/magicoder/train.py magicoder/src/magicoder/train.py
cp update/magicoder/text2code.py magicoder/experiments/text2code.py  # For fine-tuned model on DS-1000 Instruct tasks

# ===== Install dependencies =====
echo "Installing dependencies..."
pip install pdm
cd magicoder
pip install openai
pip install -U openai tiktoken transformers datasets accelerate peft bitsandbytes
pdm install


echo "✅ Setup complete."
