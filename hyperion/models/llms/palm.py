import torch
from torch import nn
from flash_attn.flash_attention import FlashMHA
from flash_attn.ops.fused_dense import FusedDenseGeluDense
from typing import Optional


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_dropout,
            bias=cfg.bias,
            batch_first=True,
            add_bias_kv=False,

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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.ByteTensor] = None):
        a = self.ln1(x)
        b, _ = self.mha(a, a, a, key_padding_mask)
        x = x + self.dropout1(b)
        c = self.ln2(x)
        d = self.ffn(c)
        x = x + self.dropout2(d)
        return x


class FusedTransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.mha = FlashMHA(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            bias=cfg.bias,
            batch_first=True,
            casual=cfg.casual,
        )

        self.ffn = FusedDenseGeluDense(
            in_features=cfg.d_model,
            hidden_features=cfg.mlp_ratio * cfg.d_model,
            bias1=False,
            bias2=False,
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
        pos = torch.arange(S, device=x.device)
        pos = self.pos_enc(pos)
        x = self.embed(x) + pos
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln(x)
        x = self.head(x)
        return x
