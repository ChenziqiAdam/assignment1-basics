import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training_results(log_path="training.log", output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the log file, skipping the configuration header
    # We find the line starting with "Step,Epoch"
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Step,Epoch"):
            header_idx = i
            break
    
    if header_idx == -1:
        print(f"Could not find header in {log_path}")
        return

    df = pd.read_csv(log_path, skiprows=header_idx)
    
    # Handle empty ValLoss (it's only logged every eval_interval)
    df['ValLoss'] = pd.to_numeric(df['ValLoss'], errors='coerce')
    
    # 1. Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Loss'], label='Train Loss', alpha=0.6)
    val_df = df.dropna(subset=['ValLoss'])
    if not val_df.empty:
        plt.plot(val_df['Step'], val_df['ValLoss'], 'ro-', label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    # 2. Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['LearningRate'])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lr.png'))
    plt.close()

    # 3. Gradient Norm
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['GradNorm'])
    plt.xlabel('Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm over Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'grad_norm.png'))
    plt.close()

    # 4. Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['TokensPerSec'])
    plt.xlabel('Step')
    plt.ylabel('Tokens/sec')
    plt.title('Training Throughput')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput.png'))
    plt.close()

    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="training.log", help="Path to training log")
    parser.add_argument("--out", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    plot_training_results(args.log, args.out)
