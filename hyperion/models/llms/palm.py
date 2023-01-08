from torch import nn
import torch
from typing import Optional


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            bias=cfg.bias,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
        )

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.register_buffer(
            'mask',
            torch.empty((cfg.max_seq_len, cfg.max_seq_len)))

        self.mask_initialized = False

    def _fill_causal_attn_mask(self):
        assert isinstance(self.mask, torch.Tensor)  # for type checking
        torch.full(size=self.mask.shape,
                   fill_value=float('-inf'),
                   out=self.mask)
        torch.triu(input=self.mask, diagonal=1, out=self.mask)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.ByteTensor] = None):
        a = self.ln1(x)
        print('mha')
        b, _ = self.mha(a, a, a, key_padding_mask)
        x = x + self.dropout1(b)
        c = self.ln2(x)
        print('ffn')
        d = self.ffn(c)
        x = x + self.dropout2(d)
        return x


class FusedTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = FlashMHA(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            attn_dropout=cfg.dropout,
            bias=cfg.bias,
            batch_first=True,
            casual=cfg.casual,
        )

        # self.ffn = FusedDenseGeluDense(
        #     in_features=cfg.d_model,
        #     hidden_features=cfg.mlp_ratio * cfg.d_model,
        #     bias1=False,
        #     bias2=False,
        # )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
        )

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.ByteTensor] = None):
        a = self.ln1(x)
        b, _ = self.mha(a, key_padding_mask)
        x = x + self.dropout1(b)
        c = self.ln2(x)
        d = self.ffn(c)
        x = x + self.dropout2(d)
        return x


class Transformer(nn.Module):
    """Some generic transformer model."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = nn.Embedding(cfg.max_len, cfg.d_model)

        if cfg.fused:
            self.blocks = nn.ModuleList(
                [FusedTransformerBlock(cfg) for _ in range(cfg.n_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
            )

        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.LongTensor, key_padding_mask: Optional[torch.ByteTensor] = None):
        _, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        print('pos enc')
        pos = self.pos_enc(pos)
        print('embed')
        x = self.embed(x) + pos
        print('blocks')
        for i, block in enumerate(self.blocks):
            print(i)
            x = block(x, key_padding_mask)
        x = self.ln(x)
        x = self.head(x)
        return x


class Config:
    dataset = 'wikitext'
    batch_size = 64
    num_workers = 2
    d_model = 768
    dropout = 0.1
    mlp_ratio = 4
    n_heads = 12
    casual = True
    bias = False
    vocab_size = 50257
    max_len = 2048
    n_layers = 12
    pin_memory = True
    lr = 5e-4
    fused = False
    tokenizer_name = 'gpt2'


cfg = Config()
model = Transformer(cfg)
x = torch.rand(24, 1024).long()
print(x)
y = model(x)
print(y.shape)
