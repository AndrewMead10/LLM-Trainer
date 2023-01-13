import torch
from torch import nn, einsum
from einops import rearrange


class BaseMHA(torch.nn.Module):
    def __init__(self, dim=512, dim_head=64, heads=8):
        super().__init__()
        # self.norm = LayerNorm(dim)
        self.heads = heads
        self.dim = dim
        self.scale = dim_head**-0.5

        self.q_lin = nn.Linear(dim, dim*3, bias=False)
        self.out_ff = nn.Linear(dim, dim, bias=False)

        self.register_buffer('mask', None, persistent=False)

    def make_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device), diagonal=1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate MHA

        Args:
            x (torch.Tensor): shape (bs, seq_len, dim_model)
        """
        seq_len = x.size(1)
        # pass throught linear layer, only 1 needed for all 3
        x = self.q_lin(x)  # bs, 3* seq_len

        # split into q,k,v
        q, k, v = x.split(self.dim, dim=-1)

        # split into heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # q * k.T
        sim = einsum('b h n j, b h m j -> b h n m', q, k)

        sim *= self.scale

        # add the mask, set masked values to -inf for the given dtype, as this will make them 0 in the softmax (e^-inf = 0)
        mask = self.make_mask(seq_len, sim.device)
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)

        # softmax across the last dimension
        sim = sim.softmax(dim=-1)

        # multiply the similaity score by v
        attn = einsum('b h n m, b h m d -> b h n d', sim, v)

        # concat so that there are no more heads
        out = rearrange(attn, 'b h n d -> b n (h d)')

        # output linear layer
        return self.out_ff(out)
