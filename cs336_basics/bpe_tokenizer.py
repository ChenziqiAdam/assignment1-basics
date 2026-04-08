"""
Byte-Pair Encoding (BPE) Tokenizer — Class-based rewrite
"""

import os
import json
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 4_096
RAW_TEXT_FILE = "tinystories_raw.txt"
MERGES_FILE   = "merges.json"
SPECIAL_SEP   = b"<|endoftext|>"

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


# ── Tokenizer Class ───────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab: dict[tuple, int], merges: list[tuple], special_tokens: dict[str, int] | None = None):
        """
        vocab          : pre-token frequency dict  {tuple_of_byte_ints: count}
        merges         : ordered list of merged pairs [(a, b), ...]
        special_tokens : optional {token_str: token_id}
        """
        self.vocab         = vocab
        self.merges        = merges
        self.special_tokens = special_tokens or {}

        # Derived lookups
        self.merge_map: dict[tuple, int] = {pair: 256 + i for i, pair in enumerate(merges)}
        self.id_to_bytes: dict[int, bytes] = self._build_id_to_bytes()

        # Add special token byte mappings
        for token_str, token_id in self.special_tokens.items():
            self.id_to_bytes[token_id] = token_str.encode("utf-8")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_id_to_bytes(self) -> dict[int, bytes]:
        id_to_bytes = {i: bytes([i]) for i in range(256)}
        for idx, pair in enumerate(self.merges, start=256):
            id_to_bytes[idx] = id_to_bytes[pair[0]] + id_to_bytes[pair[1]]
        return id_to_bytes

    def _get_pair_freqs(self, vocab: dict[tuple, int]) -> dict[tuple, int]:
        pairs: dict[tuple, int] = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return dict(pairs)

    @staticmethod
    def _merge_pair(pair: tuple[int, int], new_id: int, vocab: dict[tuple, int]) -> dict[tuple, int]:
        new_vocab: dict[tuple, int] = {}
        a, b = pair
        for word, freq in vocab.items():
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, vocab_size: int = VOCAB_SIZE) -> None:
        """Run BPE merges in-place until vocab_size is reached."""
        num_merges = vocab_size - 256
        vocab = dict(self.vocab)

        for step in range(num_merges):
            pair_freqs = self._get_pair_freqs(vocab)
            if not pair_freqs:
                print("  No more pairs -- stopping early.")
                break

            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            if pair_freqs[best_pair] < 2:
                print("  All remaining pairs have freq < 2 -- stopping early.")
                break

            new_id = 256 + step
            vocab = self._merge_pair(best_pair, new_id, vocab)
            self.merges.append(best_pair)
            self.merge_map[best_pair] = new_id
            self.id_to_bytes[new_id] = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]

            if (step + 1) % 100 == 0 or step < 5:
                print(f"  [{step+1:4d}/{num_merges}] merged {best_pair!r}  "
                      f"(freq={pair_freqs[best_pair]:,})")

        self.vocab = vocab
        print(f"Training complete: {len(self.merges)} merges, vocab size = {256 + len(self.merges)}")

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Tokenize text into token IDs, respecting special tokens."""
        # Build a pattern that splits on special tokens first
        if self.special_tokens:
            special_pat = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(f"({special_pat})", text)
        else:
            segments = [text]

        token_ids: list[int] = []
        for segment in segments:
            if segment in self.special_tokens:
                token_ids.append(self.special_tokens[segment])
                continue
            for token_str in PAT.findall(segment):
                ids = list(token_str.encode("utf-8"))
                while len(ids) > 1:
                    # Find the pair with the lowest merge index
                    pairs = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
                    pair_to_merge = None
                    min_rank = float('inf')
                    for p in pairs:
                        rank = self.merge_map.get(p, float('inf'))
                        if rank < min_rank:
                            min_rank = rank
                            pair_to_merge = p
                    
                    if pair_to_merge is None:
                        break
                    
                    # Merge ALL occurrences of pair_to_merge
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair_to_merge:
                            new_ids.append(self.merge_map[pair_to_merge])
                            i += 2
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    ids = new_ids
                token_ids.extend(ids)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a UTF-8 string."""
        raw = b"".join(self.id_to_bytes[i] for i in token_ids)
        return raw.decode("utf-8", errors="replace")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str = MERGES_FILE) -> None:
        data = {
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.merges):,} merges -> {path}")

    @classmethod
    def load(cls, path: str = MERGES_FILE, vocab: dict | None = None) -> "Tokenizer":
        with open(path) as f:
            data = json.load(f)
        merges = [tuple(p) for p in data["merges"]]
        special_tokens = data.get("special_tokens", {})
        return cls(vocab=vocab or {}, merges=merges, special_tokens=special_tokens)


# ── Standalone helpers (used by parallel workers) ─────────────────────────────

def get_pair_freqs_from(vocab: dict[tuple, int]) -> dict[tuple, int]:
    pairs: dict[tuple, int] = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return dict(pairs)


# ── Data loading / chunking ───────────────────────────────────────

def stream_to_file(max_samples: int = 10_000, path: str = RAW_TEXT_FILE) -> str:
    if os.path.exists(path):
        print(f"Found existing corpus -> {path} (skipping download)")
        return path
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    sep = SPECIAL_SEP.decode()
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            if i >= max_samples:
                break
            f.write(row["text"].strip())
            f.write(f"\n{sep}\n")
            written += 1
    print(f"Wrote {written:,} stories -> {path}")
    return path


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes = SPECIAL_SEP) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            data = file.read(mini_chunk_size)
            if not data:
                boundaries[bi] = file_size
                break
            found = data.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos + found
                break
            pos += mini_chunk_size
    return sorted(set(boundaries))


def _count_vocab_in_chunk(args: tuple) -> dict[tuple, int]:
    path, start, end = args
    local_vocab: dict[tuple, int] = defaultdict(int)
    sep = SPECIAL_SEP.decode()
    with open(path, "r", encoding="utf-8") as f:
        f.seek(start)
        text = f.read(end - start)
    for token in PAT.findall(text):
        if token == sep or not token.strip():
            continue
        local_vocab[tuple(token.encode("utf-8"))] += 1
    return dict(local_vocab)


def build_vocab_parallel(path: str, num_workers: int = 4) -> dict[tuple, int]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, SPECIAL_SEP)
    chunks = [(path, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    print(f"Building vocab across {len(chunks)} chunks ...")
    vocab: dict[tuple, int] = defaultdict(int)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for partial in ex.map(_count_vocab_in_chunk, chunks):
            for key, cnt in partial.items():
                vocab[key] += cnt
    print(f"Vocab: {len(vocab):,} unique pre-tokens")
    return dict(vocab)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_path = stream_to_file(max_samples=10_000)
    vocab    = build_vocab_parallel(raw_path, num_workers=4)

    special_tokens = {"<|endoftext|>": 256 + VOCAB_SIZE}
    tokenizer = Tokenizer(vocab=vocab, merges=[], special_tokens=special_tokens)

    print(f"\n=== Training BPE (target vocab size: {VOCAB_SIZE}) ===")
    tokenizer.train(vocab_size=VOCAB_SIZE)
    tokenizer.save()

    test = "Once upon a time there was a little girl named Lily."
    ids  = tokenizer.encode(test)
    print(f"\nInput:    {test!r}")
    print(f"Token IDs ({len(ids)}): {ids[:20]} ...")
    print(f"Decoded:  {tokenizer.decode(ids)!r}")

    loaded = Tokenizer.load()
    assert loaded.encode(test) == ids, "Round-trip failed!"
    print("Round-trip OK ✓")