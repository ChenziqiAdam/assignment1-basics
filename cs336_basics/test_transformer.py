import torch
from .transformer import transformer_lm

def test_model():
    vocab_size = 4096
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    
    model = transformer_lm(vocab_size, d_model, num_heads, d_ff, num_layers)
    idx = torch.randint(0, vocab_size, (2, 16)) # batch size 2, seq len 16
    
    output = model(idx)
    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 16, vocab_size)
    print("Transformer LM forward pass OK ✓")

if __name__ == "__main__":
    test_model()
