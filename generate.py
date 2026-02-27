import torch
import sentencepiece as spm
from config import config
from model import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor(model_file="data/bpe.model")

model = GPT(config).to(device)
checkpoint = torch.load("checkpoints/latest.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

prompt = "Artificial intelligence is"
input_ids = torch.tensor([sp.encode(prompt)], dtype=torch.long).to(device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=100)

print(sp.decode(output_ids[0].tolist()))