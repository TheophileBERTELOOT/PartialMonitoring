from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .data import MessageExample
from .text import Vocabulary


@dataclass(frozen=True)
class EncodedExample:
    text: str
    text_with_context: str
    label: int
    conversation_id: str
    message_index: int


class ModerationDataset(Dataset):
    def __init__(
        self,
        examples: list[MessageExample],
        vocab: Vocabulary | None = None,
        max_tokens: int = 96,
        max_context_tokens: int = 256,
        use_context: bool = False,
    ) -> None:
        self.examples = [
            EncodedExample(
                text=example.text,
                text_with_context=build_text_with_context(example.text, example.context, use_context),
                label=example.label,
                conversation_id=example.conversation_id,
                message_index=example.message_index,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def build_text_with_context(text: str, context: tuple[str, ...], use_context: bool) -> str:
    if not use_context or not context:
        return text
    return text + " [CTX] " + " [CTX] ".join(reversed(context))


def collate_batch(batch: list[EncodedExample]) -> dict[str, object]:
    labels: list[int] = []
    conversation_ids: list[str] = []
    message_indices: list[int] = []
    texts: list[str] = []

    for example in batch:
        texts.append(example.text_with_context)
        labels.append(example.label)
        conversation_ids.append(example.conversation_id)
        message_indices.append(example.message_index)

    return {
        "texts": texts,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "conversation_ids": conversation_ids,
        "message_indices": message_indices,
    }


class BertCollator:
    def __init__(self, tokenizer, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "right"
        self.max_length = max_length

    def __call__(self, batch: list[EncodedExample]) -> dict[str, object]:
        base = collate_batch(batch)
        if self.max_length <= self.tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("max_length must leave room for at least one message token")
        # Current message comes first; right truncation discards older context first.
        encoded = self.tokenizer(
            base["texts"], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        base.update(encoded)
        return base
