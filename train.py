import torch
from torch.cuda.amp import autocast, GradScaler
from config import config
from model import GPT
from dataset import TokenDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading dataset...")
train_dataset = TokenDataset("data/train.bin")
val_dataset = TokenDataset("data/val.bin")

print("Building model...")
model = GPT(config).to(device)

print("Total parameters:",
      sum(p.numel() for p in model.parameters()) / 1e6,
      "M")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.max_iters
)

scaler = GradScaler()


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for _ in range(50):
        xb, yb = val_dataset.get_batch(config.batch_size, device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

import os

start_iter = 0

checkpoint_path = "checkpoints/latest.pt"

if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_iter = checkpoint["iter"]
    print(f"Resumed from iteration {start_iter}")


print("Starting training...")

model.train()

for iter in range(start_iter, config.max_iters):

    optimizer.zero_grad()

    total_loss = 0

    for _ in range(config.grad_accum_steps):

        xb, yb = train_dataset.get_batch(config.batch_size, device)

        with autocast():
            logits, loss = model(xb, yb)
            loss = loss / config.grad_accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if iter % 100 == 0:
        print(f"Iter {iter} | Train Loss {total_loss:.4f}")

    if iter % 1000 == 0:
        val_loss = evaluate()
        print(f"Validation Loss: {val_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter
        }, "checkpoints/latest.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter
        }, f"checkpoints/ckpt_{iter}.pt")