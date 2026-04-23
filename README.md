# Infinite Shakespeare LLM

A character-level language modeling project built with PyTorch, trained on the Tiny Shakespeare dataset.

This repository contains two scripts:
- `shakespear.py`: a step-by-step educational walkthrough of tokenization, batching, and a basic bigram model.
- `bigram.py`: a transformer-style character model with multi-head self-attention, feed-forward blocks, layer normalization, and autoregressive text generation.

## Features

- Character-level tokenization and encoding/decoding utilities
- Train/validation split and randomized mini-batch sampling
- Causal self-attention with masking for autoregressive modeling
- Configurable model hyperparameters (layers, heads, embedding size, dropout, context length)
- GPU acceleration support (CUDA) when available
- Text generation from a learned Shakespeare-style distribution

## Project Structure

```text
.
|-- bigram.py         # Transformer-style language model training + generation
|-- shakespear.py     # Educational baseline + bigram walkthrough
|-- input.txt         # Tiny Shakespeare training corpus
|-- requirements.txt  # Python dependencies
`-- .gitignore
