import torch
from einops import rearrange


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implementation."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                # Apply weight decay
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                # Update parameters
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(group['eps']), value=-step_size)


def learning_rate_scheduler(optimizer, step, warmup_steps=1000):
    """Learning rate scheduler with linear warmup and inverse square root decay."""
    if step < warmup_steps:
        lr_scale = step / warmup_steps
    else:
        lr_scale = (warmup_steps ** 0.5) / (step ** 0.5)
    for param_group in optimizer.param_groups:
        initial_lr = param_group.get('initial_lr', param_group['lr'])
        param_group['lr'] = initial_lr * lr_scale

def gradient_clipping(parameters, max_norm):
    """Clip gradients to a maximum norm."""
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for language modeling."""
    batch_size, seq_len, vocab_size = logits.shape
    logits = rearrange(logits, "b s v -> (b s) v")
    targets = rearrange(targets, "b s -> (b s)")
    log_probs = torch.log_softmax(logits, dim=-1)
    return -log_probs[torch.arange(batch_size * seq_len), targets].mean()