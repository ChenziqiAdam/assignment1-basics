import torch
from .transformer import transformer_lm, generate
from .bpe_tokenizer import Tokenizer

def run_eval(prompt: str, model_path: str, merges_path: str):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    tokenizer = Tokenizer.load(merges_path)
    
    # 3. Initialize and Load Model
    # Note: These parameters should match the ones used in run_training.py
    model_config = {
        "vocab_size": 256 + len(tokenizer.merges) + 1,
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "num_layers": 6
    }
    model = transformer_lm(**model_config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Tokenize Prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    # 5. Generate
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    output_ids = generate(model, input_ids, max_length=50)
    
    # 6. Decode and Print
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\nGenerated Output:\n{generated_text}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained Transformer model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--model", type=str, default="transformer_final.pt", help="Path to model checkpoint")
    parser.add_argument("--merges", type=str, default="cs336_basics/merges.json", help="Path to tokenizer merges")
    
    args = parser.parse_args()
    run_eval(args.prompt, args.model, args.merges)
