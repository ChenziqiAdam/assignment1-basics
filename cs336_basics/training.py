import torch
from .optimizer import learning_rate_scheduler, gradient_clipping

def data_loading(x: torch.Tensor, batch_size: int, context_length: int, device: torch.device):
    """Prepare batches of input and target sequences for training."""
    num_batches = x.size(0) // batch_size
    x = x[:num_batches * batch_size].view(batch_size, -1).to(device)
    for i in range(0, x.size(1) - context_length, context_length):
        inputs = x[:, i:i + context_length]
        targets = x[:, i + 1:i + context_length + 1]
        yield inputs, targets

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    """Save model and optimizer state to a checkpoint file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def training_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader_fn, data_args, num_epochs: int, device: torch.device, warmup_steps=1000, max_norm=1.0, log_interval=100, log_path="training.log"):
    """Main training loop for the transformer model with detailed logging."""
    model.to(device)
    global_step = 0
    
    # Initialize log file
    with open(log_path, 'w') as f:
        f.write("Step,Epoch,Loss,LearningRate\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_steps = 0
        # data_loader_fn is expected to be data_loading
        for inputs, targets in data_loader_fn(*data_args, device=device):
            global_step += 1
            num_steps += 1
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            gradient_clipping(model.parameters(), max_norm=max_norm)
            
            # LR Scheduler
            learning_rate_scheduler(optimizer, global_step, warmup_steps=warmup_steps)
            
            # Get current LR for logging
            current_lr = optimizer.param_groups[0]['lr']
            
            optimizer.step()
            total_loss += loss.item()

            if global_step % log_interval == 0:
                print(f'Step {global_step} (Epoch {epoch+1}), Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
                with open(log_path, 'a') as f:
                    f.write(f'{global_step},{epoch+1},{loss.item():.4f},{current_lr:.2e}\n')
            
        if num_steps > 0:
            avg_loss = total_loss / num_steps
            print(f'--- Epoch {epoch + 1}/{num_epochs} Complete, Avg Loss: {avg_loss:.4f} ---')

def experiment_logging(epoch: int, loss: float, log_path: str):
    """Log training metrics to a file."""
    with open(log_path, 'a') as f:
        f.write(f'Epoch {epoch}, Loss: {loss:.4f}\n')