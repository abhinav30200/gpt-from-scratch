class Config:
    # Data
    vocab_size = 20000
    block_size = 256

    # Model
    n_embd = 384
    n_layer = 6
    n_head = 6
    dropout = 0.1

    # Training
    batch_size = 8
    grad_accum_steps = 4
    learning_rate = 3e-4
    weight_decay = 0.1
    max_iters = 200000
    warmup_iters = 2000

config = Config()