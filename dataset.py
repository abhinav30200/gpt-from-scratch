import numpy as np
import torch
from config import config


class TokenDataset:
    def __init__(self, path):
        # Memory-map the file (does NOT load full file into RAM)
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = config.block_size

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size, device):
        # Random start positions
        ix = torch.randint(
            0,
            len(self.data) - self.block_size - 1,
            (batch_size,)
        )

        x = torch.stack([
            torch.from_numpy(
                self.data[i:i+self.block_size].astype(np.int64)
            )
            for i in ix
        ])

        y = torch.stack([
            torch.from_numpy(
                self.data[i+1:i+self.block_size+1].astype(np.int64)
            )
            for i in ix
        ])

        return x.to(device), y.to(device)