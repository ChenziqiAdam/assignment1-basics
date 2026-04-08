import torch
import torch.nn as nn
from .transformer import transformer_lm
from .optimizer import AdamW
from .training import training_loop, data_loading

def test_training():
    vocab_size = 100
    d_model = 32
    num_heads = 2
    d_ff = 64
    num_layers = 1
    
    model = transformer_lm(vocab_size, d_model, num_heads, d_ff, num_layers)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Mock data
    x = torch.randint(0, vocab_size, (1000,))
    
    device = torch.device("cpu")
    
    # data_loading(x, batch_size, context_length, device)
    batch_size = 4
    context_length = 8
    
    print("Starting training loop test...")
    training_loop(
        model, 
        optimizer, 
        data_loading, 
        (x, batch_size, context_length), 
        num_epochs=2, 
        device=device,
        warmup_steps=10
    )
    print("Training loop test OK ✓")

if __name__ == "__main__":
    test_training()
