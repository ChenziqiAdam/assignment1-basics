import os
import sys
import math
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Add the project root to the python path so that we can import cs336_basics
sys.path.append(str(Path(__file__).parent.parent))

from cs336_basics.modules import (
    Embedding,
    Linear,
    MultiHeadAttention,
    RMSNorm,
    RoPE,
    SwiGLU,
    scaled_dot_product_attention,
    softmax,
)
from cs336_basics.optimizer import AdamW, cross_entropy_loss, gradient_clipping
from cs336_basics.transformer import transformer_block, transformer_lm
from cs336_basics.training import save_checkpoint, load_checkpoint
from cs336_basics.bpe_tokenizer import Tokenizer, PAT


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    return F.linear(in_features, weights)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return F.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    gate = F.silu(F.linear(in_features, w1_weight))
    value = F.linear(in_features, w3_weight)
    return F.linear(gate * value, w2_weight)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.modules import rearrange
    q = rearrange(F.linear(in_features, q_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    k = rearrange(F.linear(in_features, k_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    v = rearrange(F.linear(in_features, v_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    seq_len = in_features.shape[-2]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).view(1, 1, seq_len, seq_len)
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=mask)
    attn_output = rearrange(attn_output, "... h s d_k -> ... s (h d_k)")
    return F.linear(attn_output, o_proj_weight)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.modules import rearrange
    q = rearrange(F.linear(in_features, q_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    k = rearrange(F.linear(in_features, k_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    v = rearrange(F.linear(in_features, v_proj_weight), "... s (h d_k) -> ... h s d_k", h=num_heads)
    rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
    seq_len = in_features.shape[-2]
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).expand(in_features.shape[:-2] + (seq_len,))
    q_rotated = rope(q, token_positions)
    k_rotated = rope(k, token_positions)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).view(1, 1, seq_len, seq_len)
    attn_output = run_scaled_dot_product_attention(q_rotated, k_rotated, v, mask=mask)
    attn_output = rearrange(attn_output, "... h s d_k -> ... s (h d_k)")
    return F.linear(attn_output, o_proj_weight)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    h = in_features
    seq_len = in_features.shape[1]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).view(1, 1, seq_len, seq_len)
    positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(in_features.shape[0], -1)
    ln1_w = weights["ln1.weight"]
    ln1_out = F.rms_norm(h, (d_model,), ln1_w, eps=1e-5)
    from cs336_basics.modules import rearrange
    q = rearrange(F.linear(ln1_out, weights["attn.q_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
    k = rearrange(F.linear(ln1_out, weights["attn.k_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
    v = rearrange(F.linear(ln1_out, weights["attn.v_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
    rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
    q = rope(q, positions); k = rope(k, positions)
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=mask)
    attn_output = rearrange(attn_output, "b h s d_k -> b s (h d_k)")
    attn_output = F.linear(attn_output, weights["attn.output_proj.weight"])
    h = h + attn_output
    ln2_w = weights["ln2.weight"]
    ln2_out = F.rms_norm(h, (d_model,), ln2_w, eps=1e-5)
    ffn_w1 = weights["ffn.w1.weight"]; ffn_w2 = weights["ffn.w2.weight"]; ffn_w3 = weights["ffn.w3.weight"]
    ffn_out = F.linear(F.silu(F.linear(ln2_out, ffn_w1)) * F.linear(ln2_out, ffn_w3), ffn_w2)
    h = h + ffn_out
    return h


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    token_emb_w = weights["token_embeddings.weight"]
    ln_final_w = weights["ln_final.weight"]
    lm_head_w = weights["lm_head.weight"]
    h = F.embedding(in_indices, token_emb_w)
    seq_len = in_indices.shape[1]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_indices.device)).view(1, 1, seq_len, seq_len)
    positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(in_indices.shape[0], -1)
    for i in range(num_layers):
        ln1_w = weights[f"layers.{i}.ln1.weight"]
        ln1_out = F.rms_norm(h, (d_model,), ln1_w, eps=1e-5)
        from cs336_basics.modules import rearrange
        q = rearrange(F.linear(ln1_out, weights[f"layers.{i}.attn.q_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
        k = rearrange(F.linear(ln1_out, weights[f"layers.{i}.attn.k_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
        v = rearrange(F.linear(ln1_out, weights[f"layers.{i}.attn.v_proj.weight"]), "b s (h d_k) -> b h s d_k", h=num_heads)
        rope = RoPE(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length)
        q = rope(q, positions); k = rope(k, positions)
        attn_out = run_scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = rearrange(attn_out, "b h s d_k -> b s (h d_k)")
        attn_out = F.linear(attn_out, weights[f"layers.{i}.attn.output_proj.weight"])
        h = h + attn_out
        ln2_w = weights[f"layers.{i}.ln2.weight"]
        ln2_out = F.rms_norm(h, (d_model,), ln2_w, eps=1e-5)
        ffn_w1 = weights[f"layers.{i}.ffn.w1.weight"]; ffn_w2 = weights[f"layers.{i}.ffn.w2.weight"]; ffn_w3 = weights[f"layers.{i}.ffn.w3.weight"]
        ffn_out = F.linear(F.silu(F.linear(ln2_out, ffn_w1)) * F.linear(ln2_out, ffn_w3), ffn_w2)
        h = h + ffn_out
    h = F.rms_norm(h, (d_model,), ln_final_w, eps=1e-5)
    return F.linear(h, lm_head_w)


def run_rmsnorm(d_model, eps, weights, in_features):
    return F.rms_norm(in_features, (d_model,), weights, eps=eps)


def run_silu(in_features):
    return F.silu(in_features)


def run_get_batch(dataset, batch_size, context_length, device):
    import numpy as np
    ix = np.random.randint(0, len(dataset) - context_length, batch_size)
    x = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[i + 1 : i + context_length + 1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def run_softmax(in_features, dim):
    return F.softmax(in_features, dim=dim)


def run_cross_entropy(inputs, targets):
    return F.cross_entropy(inputs, targets)


def run_gradient_clipping(parameters, max_l2_norm):
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls():
    return AdamW


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters: return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters: return min_learning_rate
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(model, optimizer, iteration, out):
    torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, out)


def run_load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('iteration', checkpoint.get('epoch', 0))


def get_tokenizer(vocab, merges, special_tokens=None):
    class TokenizerWrapper:
        def __init__(self, vocab, merges, special_tokens_list):
            self.vocab = vocab
            self.merges = merges
            self.special_tokens = special_tokens_list or []
            self.byte_to_id = {v: k for k, v in vocab.items()}
            self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
            self.special_to_id = {st: self.byte_to_id[st.encode("utf-8")] for st in self.special_tokens if st.encode("utf-8") in self.byte_to_id}

        def encode(self, text: str) -> list[int]:
            if not text: return []
            special_pat = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(f"({special_pat})", text) if special_pat else [text]
            token_ids = []
            for segment in segments:
                if segment in self.special_to_id: token_ids.append(self.special_to_id[segment]); continue
                if not segment: continue
                for word in PAT.findall(segment):
                    ids = [self.byte_to_id[bytes([b])] for b in word.encode("utf-8")]
                    while len(ids) > 1:
                        best_pair, min_rank = None, float('inf')
                        for i in range(len(ids) - 1):
                            pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                            rank = self.merge_ranks.get(pair, float('inf'))
                            if rank < min_rank: min_rank, best_pair = rank, pair
                        if best_pair is None: break
                        new_ids, i = [], 0
                        while i < len(ids):
                            if i < len(ids) - 1 and (self.vocab[ids[i]], self.vocab[ids[i+1]]) == best_pair:
                                new_ids.append(self.byte_to_id[self.vocab[ids[i]] + self.vocab[ids[i+1]]]); i += 2
                            else: new_ids.append(ids[i]); i += 1
                        ids = new_ids
                    token_ids.extend(ids)
            return token_ids

        def encode_iterable(self, iterable):
            for text in iterable: yield from self.encode(text)

        def decode(self, ids):
            return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    return TokenizerWrapper(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split text on special tokens to avoid merging them
    special_pat = "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    if special_pat:
        segments = re.split(f"({special_pat})", text)
        train_text_list = [s for s in segments if s not in special_tokens]
    else:
        train_text_list = [text]

    word_freqs = Counter()
    for segment in train_text_list:
        word_freqs.update(PAT.findall(segment))
        
    word_tokens = {word: [bytes([b]) for b in word.encode("utf-8")] for word in word_freqs}
    pair_freqs = Counter()
    pair_to_words = defaultdict(set)
    for word, tokens in word_tokens.items():
        freq = word_freqs[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1]); pair_freqs[pair] += freq; pair_to_words[pair].add(word)

    words_in_order = []
    seen_words = set()
    for segment in train_text_list:
        for word in PAT.findall(segment):
            if word not in seen_words: words_in_order.append(word); seen_words.add(word)
            
    pair_first_seen = {}; p_idx = 0
    for word in words_in_order:
        tokens = word_tokens[word]
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            if pair not in pair_first_seen: pair_first_seen[pair] = p_idx; p_idx += 1

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        if not pair_freqs: break
        best_pair = None; max_f = -1
        for p, f in pair_freqs.items():
            if f > max_f: max_f = f; best_pair = p
            elif f == max_f:
                if best_pair is None or pair_first_seen[p] < pair_first_seen[best_pair]: best_pair = p
        if best_pair is None or max_f < 1: break
        merges.append(best_pair); new_token = best_pair[0] + best_pair[1]
        affected_words = pair_to_words[best_pair]
        for word in list(affected_words):
            old_tokens = word_tokens[word]; new_tokens = []
            i = 0
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i+1]) == best_pair:
                    new_tokens.append(new_token); i += 2
                else: new_tokens.append(old_tokens[i]); i += 1
            freq = word_freqs[word]
            for i in range(len(old_tokens) - 1):
                p = (old_tokens[i], old_tokens[i+1]); pair_freqs[p] -= freq
            word_tokens[word] = new_tokens
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i+1])
                pair_freqs[p] += freq; pair_to_words[p].add(word)
                if p not in pair_first_seen: pair_first_seen[p] = p_idx; p_idx += 1
        pair_to_words[best_pair] = set()
        # Clean up pair_freqs only when they are the best_pair to keep it fast
        if best_pair in pair_freqs: del pair_freqs[best_pair]

    byte_to_id = {bytes([i]): i for i in range(256)}
    for i, p in enumerate(merges): byte_to_id[p[0] + p[1]] = 256 + i
    curr_id = 256 + len(merges)
    for st in special_tokens: byte_to_id[st.encode("utf-8")] = curr_id; curr_id += 1
    return {v: k for k, v in byte_to_id.items()}, merges
