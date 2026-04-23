"""Run 4: Long-context handling strategies."""

# Head-tail packing is implemented in src/data/dataset.py (HeadTailCodeDataset).
# This module provides sliding-window chunk aggregation as an alternative.

import torch
import torch.nn as nn
import numpy as np


def sliding_window_inference(
    model: nn.Module,
    tokenizer,
    text: str,
    chunk_size: int = 384,
    overlap: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference with overlapping sliding windows and aggregate logits."""
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = tokens["input_ids"].squeeze(0)

    if len(input_ids) <= chunk_size:
        encoding = tokenizer(text, max_length=chunk_size, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            out = model(**encoding)
        return torch.softmax(out["logits"], dim=-1).cpu().numpy()[0]

    stride = chunk_size - overlap
    all_logits = []

    for start in range(0, len(input_ids), stride):
        chunk = input_ids[start : start + chunk_size]
        if len(chunk) < 64:  # skip very short trailing chunks
            break
        attention_mask = torch.ones(len(chunk), dtype=torch.long)

        # Pad if needed
        if len(chunk) < chunk_size:
            pad_len = chunk_size - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        batch = {
            "input_ids": chunk.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            out = model(**batch)
        all_logits.append(out["logits"].cpu())

    # Aggregate: mean of softmax probabilities
    all_probs = [torch.softmax(l, dim=-1) for l in all_logits]
    mean_probs = torch.stack(all_probs).mean(dim=0)
    return mean_probs.numpy()[0]
