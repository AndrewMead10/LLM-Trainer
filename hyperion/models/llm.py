import torch
from torch import nn

from modules.attention import BaseMHA


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.attn = BaseMHA(dim, dim_head, heads)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(self.norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads,
                             cfg.d_model//cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return x


x = torch.randint(0, 10000, (64, 512))

cfg = {
    'vocab_size': 10000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'max_len': 512,
    'dropout': 0.1
}

# convert the above dict to a class
cfg = type('Config', (object,), cfg)

model = Transformer(cfg)
print('num_params: ', sum(p.numel()
      for p in model.parameters() if p.requires_grad))
y = model(x)
print(y.shape)
