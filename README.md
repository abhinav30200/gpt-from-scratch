# GPT-Style Transformer Language Model (Built From Scratch)

A decoder-only Transformer language model implemented and pretrained entirely from first principles in PyTorch.  
The model was trained from random initialization on ~493M tokens of web-scale text and achieves a validation perplexity of ~49.5.

This project demonstrates end-to-end large language model engineering, including tokenizer training, scalable data preprocessing, stable mixed-precision training, evaluation, and optimized autoregressive inference.

---

## 🚀 Overview

This repository contains a complete implementation of a GPT-style autoregressive language model, including:

- Transformer architecture (masked multi-head self-attention)
- Custom subword tokenizer (BPE via SentencePiece)
- Memory-efficient data pipeline
- Mixed precision (AMP) training
- Cosine learning rate scheduling
- Gradient clipping
- Checkpoint resumption
- Perplexity-based evaluation
- Advanced decoding (temperature, top-k, nucleus sampling)

The model was trained on approximately 493 million tokens derived from web-scale text corpora.

---

## 🧠 Model Architecture

- Architecture: Decoder-only Transformer
- Parameters: ~26M
- Vocabulary size: 20,000 (custom BPE)
- Context length: 256 tokens
- Number of layers: 6
- Attention heads: 6
- Embedding dimension: 384
- Activation: ReLU
- Normalization: LayerNorm
- Positional encoding: Learned embeddings

This architecture follows the GPT family design (causal masking + autoregressive training).

---

## 📊 Training Details

- Training from random initialization
- Dataset size: ~493M tokens
- Mixed precision training (torch.cuda.amp)
- Cosine annealing learning rate schedule
- Gradient norm clipping for stability
- Fault-tolerant checkpointing and resume support

### Final Metrics

- Validation Loss: ~3.90
- Validation Perplexity: ~49.5

Perplexity is computed as:


perplexity = exp(cross_entropy_loss)


---

## 🧩 Tokenization

A 20K-vocabulary Byte Pair Encoding (BPE) tokenizer was trained using SentencePiece.

- Subword tokenization
- Memory-mapped binary dataset storage
- Efficient batched loading for large-scale training

---

## ✨ Inference

Autoregressive generation supports:

- Temperature scaling
- Top-k sampling
- Nucleus (top-p) sampling
- Repetition penalty

These techniques significantly improve coherence and reduce degeneration during long-sequence generation.

---

## 📁 Project Structure


├── src/
│ ├── model.py
│ ├── train.py
│ ├── generate.py
│ ├── config.py
│
├── data/ # Excluded (large datasets)
├── checkpoints/ # Excluded (model weights)
├── README.md
└── requirements.txt


Large dataset files and model checkpoints are excluded due to repository size limits.

---

## 🛠 Setup

```bash
pip install -r requirements.txt
Training
python src/train.py
Generation
python src/generate.py
⚠ Limitations

Model size (26M parameters) limits deep semantic reasoning

Trained on subset of web corpus (not full-scale GPT-2 data)

Designed for research and educational large language model engineering

🎯 Key Takeaways

This project demonstrates:

Implementation of a Transformer architecture from scratch

Large-scale language model pretraining

Training stability techniques used in modern LLMs

Proper quantitative evaluation using perplexity

Practical autoregressive inference optimizations

📌 Future Work

Instruction fine-tuning

Model scaling experiments

Distributed training

Quantization for deployment

Cloud-based inference serving

📜 License

For educational and research purposes.
