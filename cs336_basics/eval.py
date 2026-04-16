import torch
import math
from .transformer import transformer_lm, generate
from .bpe_tokenizer import Tokenizer
from .training import evaluate, data_loading

def run_eval(prompt: str, model_path: str, merges_path: str, test_file: str = None):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    tokenizer = Tokenizer.load(merges_path)
    
    # 3. Initialize and Load Model
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Optional: Calculate Metrics on Test File
    if test_file:
        print(f"--- Evaluating on {test_file} ---")
        with open(test_file, "r", encoding="utf-8") as f:
            test_text = f.read()
        test_tokens = tokenizer.encode(test_text)
        test_data = torch.tensor(test_tokens, dtype=torch.long)
        
        batch_size = 32
        context_length = 128
        test_loss = evaluate(model, data_loading, (test_data, batch_size, context_length), device)
        perplexity = math.exp(test_loss)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Perplexity: {perplexity:.4f}")
        
        with open("eval_results.txt", "a") as f:
            f.write(f"Model: {model_path}, Test File: {test_file}, Loss: {test_loss:.4f}, Perplexity: {perplexity:.4f}\n")

    # 5. Tokenize Prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    # 6. Generate
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    output_ids = generate(model, input_ids, max_length=50)
    
    # 7. Decode and Print
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\nGenerated Output:\n{generated_text}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained Transformer model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--model", type=str, default="transformer_final.pt", help="Path to model checkpoint")
    parser.add_argument("--merges", type=str, default="cs336_basics/merges.json", help="Path to tokenizer merges")
    parser.add_argument("--test_file", type=str, default=None, help="Path to a test text file for loss calculation")
    
    args = parser.parse_args()
    run_eval(args.prompt, args.model, args.merges, args.test_file)
