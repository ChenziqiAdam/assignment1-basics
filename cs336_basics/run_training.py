import torch
import os
from .bpe_tokenizer import Tokenizer, stream_to_file, build_vocab_parallel, VOCAB_SIZE
from .transformer import transformer_lm
from .optimizer import AdamW
from .training import training_loop, data_loading, save_checkpoint

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Tokenizer Training/Loading
    # Use paths relative to the script's directory
    script_dir = os.path.dirname(__file__)
    raw_path = os.path.join(script_dir, "tinystories_raw.txt")
    merges_path = os.path.join(script_dir, "merges.json")
    
    if not os.path.exists(merges_path):
        print("--- Training Tokenizer ---")
        stream_to_file(max_samples=10000, path=raw_path)
        vocab = build_vocab_parallel(raw_path, num_workers=4)
        special_tokens = {"<|endoftext|>": VOCAB_SIZE}
        tokenizer = Tokenizer(vocab=vocab, merges=[], special_tokens=special_tokens)
        tokenizer.train(vocab_size=VOCAB_SIZE)
        tokenizer.save(merges_path)
    else:
        print("--- Loading Tokenizer ---")
        tokenizer = Tokenizer.load(merges_path)

    # 3. Prepare Data (Tokenize full corpus)
    print("--- Tokenizing Dataset ---")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"Dataset size: {len(data_tensor):,} tokens")

    # 4. Initialize Model
    print("--- Initializing Model ---")
    # Determine vocab size from tokenizer
    max_token_id = max(tokenizer.id_to_bytes.keys())
    model_config = {
        "vocab_size": max_token_id + 1,
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "num_layers": 6
    }
    model = transformer_lm(**model_config).to(device)
    
    # 5. Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

    # 6. Run Training
    print("--- Starting Training ---")
    batch_size = 32
    context_length = 128
    num_epochs = 5
    warmup_steps = 500
    log_interval = 100
    log_path = "training.log"

    # Log full configuration
    config_msg = (
        f"--- Training Config ---\n"
        f"Device: {device}\n"
        f"Model: {model_config}\n"
        f"Batch Size: {batch_size}\n"
        f"Context Length: {context_length}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Initial LR: 5e-4\n"
        f"Weight Decay: 0.1\n"
        f"Warmup Steps: {warmup_steps}\n"
        f"Dataset Size: {len(data_tensor):,} tokens\n"
        f"------------------------"
    )
    print(config_msg)
    with open(log_path, 'w') as f:
        f.write(config_msg + "\n\n")

    training_loop(
        model=model,
        optimizer=optimizer,
        data_loader_fn=data_loading,
        data_args=(data_tensor, batch_size, context_length),
        num_epochs=num_epochs,
        device=device,
        warmup_steps=warmup_steps,
        log_interval=log_interval,
        log_path=log_path
    )

    # 7. Save Final Model
    save_checkpoint(model, optimizer, num_epochs, "transformer_final.pt")
    print("Training complete. Model saved to transformer_final.pt")

if __name__ == "__main__":
    main()
