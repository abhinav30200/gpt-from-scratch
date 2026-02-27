import os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# ===== CONFIG =====
OUTPUT_DIR = "data"
RAW_TEXT_FILE = os.path.join(OUTPUT_DIR, "openwebtext.txt")
TOKENIZER_PREFIX = os.path.join(OUTPUT_DIR, "bpe")
VOCAB_SIZE = 20000
TRAIN_RATIO = 0.9

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading OpenWebText subset...")

# Use only 5% to keep things manageable
dataset = load_dataset("openwebtext", split="train[:5%]")

print("Writing raw text file...")

with open(RAW_TEXT_FILE, "w", encoding="utf-8") as f:
    for example in tqdm(dataset):
        text = example["text"].strip()
        if text:
            f.write(text + "\n")

print("Training BPE tokenizer...")

spm.SentencePieceTrainer.train(
    input=RAW_TEXT_FILE,
    model_prefix=TOKENIZER_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0
)

print("Loading tokenizer...")
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PREFIX + ".model")

print("Encoding dataset...")

tokens = []

with open(RAW_TEXT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ids = sp.encode(line.strip())
        tokens.extend(ids)

tokens = np.array(tokens, dtype=np.uint16)

print(f"Total tokens: {len(tokens):,}")

split = int(len(tokens) * TRAIN_RATIO)
train_tokens = tokens[:split]
val_tokens = tokens[split:]

train_tokens.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
val_tokens.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

print("Done. Saved train.bin and val.bin.")