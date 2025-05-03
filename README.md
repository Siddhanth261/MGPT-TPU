# Synthesis — TPU-Optimized MiniGPT in JAX

Synthesis is a research-grade ML systems project designed to demonstrate mastery of:

- JAX + XLA compilation
- Google TPU architecture
- Data-parallel training using `pmap` and (later) `pjit`
- Building a GPT-style transformer model from scratch
- Designing a custom TPU-oriented training engine

This project includes three major components:

## 1️⃣ Synthesis Core — The TPU Training Framework
A custom-built JAX-based training engine featuring:
- TPU device discovery
- Automatic sharding and data parallelism
- AdamW optimizer
- Checkpointing system
- JSON logging
- Configurable training loops

## 2️⃣ Synthesis GPT — A MiniGPT Transformer
A decoder-only transformer with:
- Token + positional (or rotary) embeddings
- Multi-head causal self-attention
- Feed-forward layers (MLP + GELU)
- LayerNorm + residual connections
- Weight tying
- Generation utilities (sampling, top-k, top-p)

## 3️⃣ Data Pipeline
A fast TPU-ready text pipeline:
- BPE tokenizer
- Sharded dataset preparation
- Sequence chunking + random sampling
- Prefetch + device dispatch

---