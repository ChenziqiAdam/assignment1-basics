import torch
from .modules import Linear, Embedding, RMSNorm, SwiGLU, MultiHeadAttention


class transformer_block(torch.nn.Module):
    """A single pre-norm transformer block with multi-head attention and feedforward network."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm multi-head attention
        x = x + self.attn(self.ln1(x), mask=mask)
        # Pre-norm feedforward network
        x = x + self.ffn(self.ln2(x))
        return x
    

class transformer_lm(torch.nn.Module):
    """A transformer-based language model with pre-norm blocks."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([transformer_block(d_model, num_heads, d_ff, theta=theta, max_seq_len=max_seq_len) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        # Create causal mask: [batch, 1, seq_len, seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        
        x = self.token_embeddings(idx)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_final(x)
        return self.lm_head(x)
    

def generate(model: torch.nn.Module, idx: torch.Tensor, max_length: int) -> torch.Tensor:
    """Generate text by autoregressively sampling from the model."""
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(idx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
    return idx