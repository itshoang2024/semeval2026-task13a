"""PyTorch Dataset for code classification."""

import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class HeadTailCodeDataset(Dataset):
    """Dataset that keeps head + tail tokens for long code sequences."""

    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, head_tokens: int = 256, tail_tokens: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens
        self.max_length = head_tokens + tail_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)

        if len(input_ids) <= self.max_length:
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            head = input_ids[: self.head_tokens]
            tail = input_ids[-self.tail_tokens :]
            combined = torch.cat([head, tail])

            attention_mask = torch.ones(self.max_length, dtype=torch.long)
            encoding = {"input_ids": combined, "attention_mask": attention_mask}

        item = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
