"""
Parameter Initialization

Linear weights: N(mu=0, sigma^2=2/(d_in+d_out)) truncated at [-3sigma, 3sigma]
Embedding:      N(mu=0, sigma^2=1) truncated at [-3, 3]
RMSNorm:        1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce


def _truncated_normal_(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Fill tensor with truncated normal N(mean, std^2), clipped at [-3std, 3std]."""
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-3 * std, b=3 * std)
    return tensor


class Linear(nn.Module):
    """Linear layer (no bias) with Xavier-style truncated normal initialization."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        _truncated_normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class Embedding(nn.Module):
    """Token embedding with truncated normal initialization."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        _truncated_normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.d_model,), self.weight, eps=self.eps)


class SwiGLU(nn.Module):
    """
    Position-wise feed-forward network with SwiGLU activation.

    SwiGLU(x) = SiLU(W1 x) * (W3 x)
    Output:     W2 @ SwiGLU(x)

    Typically d_ff = 8/3 * d_model (rounded to a multiple of 64).
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model] -> [..., d_model]
        gate = F.silu(self.w1(x))   # [..., d_ff]
        value = self.w3(x)          # [..., d_ff]
        return self.w2(gate * value)


class RoPE(nn.Module):
    """RoPE: Rotary positional embedding."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute angles via explicit outer product: [max_seq_len, d_k//2]
        seq_idx = torch.arange(max_seq_len, device=device)
        inv_freqs = torch.pow(theta, -torch.arange(0, d_k, 2, device=device) / d_k)
        angles = einsum(seq_idx, inv_freqs, "i, j -> i j")

        self.register_buffer("cos_emb", torch.cos(angles))
        self.register_buffer("sin_emb", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x:               [..., seq_len, d_k]
        # token_positions: [..., seq_len]
        # returns:         [..., seq_len, d_k]
        cos = self.cos_emb[token_positions].unsqueeze(-3)  # [..., 1, seq_len, d_k//2]
        sin = self.sin_emb[token_positions].unsqueeze(-3)  # [..., 1, seq_len, d_k//2]

        # Split d_k into consecutive pairs: [..., seq_len, d_k//2, 2]
        x_pairs = rearrange(x[..., :self.d_k], "... (d r) -> ... d r", r=2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

        # Apply rotation and merge pairs back
        x_rotated = rearrange(
            torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1),
            "... d r -> ... (d r)",
        )
        return torch.cat([x_rotated, x[..., self.d_k:]], dim=-1)
    

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """Scaled dot-product attention."""
    # q: [b, h, s_q, d_k], k: [b, h, s_k, d_k], v: [b, h, s_k, d_v]
    d_k = q.shape[-1]
    # [b, h, s_q, s_k]
    scores = einsum(q, k, "b h s_q d, b h s_k d -> b h s_q s_k") / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    # [b, h, s_q, d_v]
    return einsum(attn_weights, v, "b h s_q s_k, b h s_k d -> b h s_q d")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and scaled dot-product attention."""

    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape

        q = rearrange(self.q_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        q_rotated = self.rope(q, positions)
        k_rotated = self.rope(k, positions)

        attn_output = scaled_dot_product_attention(q_rotated, k_rotated, v, mask=mask)
        attn_output = rearrange(attn_output, "b h s d_k -> b s (h d_k)")
        return self.output_proj(attn_output)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Linear
    linear = Linear(3, 2)
    x = torch.randn(4, 3)
    print("Linear output:", linear(x))

    # Embedding
    emb = Embedding(10, 4)
    idx = torch.tensor([0, 3, 7])
    print("Embedding output:", emb(idx).shape)

    # RMSNorm
    norm = RMSNorm(4)
    print("RMSNorm output:", norm(emb(idx)).shape)

    # SwiGLU
    ffn = SwiGLU(d_model=4, d_ff=11)  # 8/3 * 4 ≈ 11
    print("SwiGLU output:", ffn(emb(idx)).shape)

    # RoPE
    rope = RoPE(theta=10000.0, d_k=4, max_seq_len=16)
    q = torch.randn(2, 1, 8, 4)  # [batch, heads, seq_len, d_k]
    positions = torch.arange(8).unsqueeze(0).expand(2, -1)
    print("RoPE output:", rope(q, positions).shape)