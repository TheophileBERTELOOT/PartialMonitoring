import torch
from transformers import BertTokenizer

from partial_monitoring.data import MessageExample, load_conversations, split_by_conversation
from partial_monitoring.dataset import BertCollator, ModerationDataset


def make_tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "current", "old", "recent", "long", "ctx", "[", "]"]))
    return BertTokenizer(str(vocab), do_lower_case=True)


def test_collator_preserves_current_message_even_with_left_truncation(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    tokenizer.truncation_side = "left"
    example = MessageExample("one", 1, "current " * 20, 1, ("old " * 100,))
    dataset = ModerationDataset([example], use_context=True)
    batch = BertCollator(tokenizer, max_length=8)([dataset[0]])
    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    assert tokens == ["[CLS]"] + ["current"] * 6 + ["[SEP]"]
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (1, 8)
    assert batch["labels"].dtype == torch.float32


def test_collator_fills_remaining_budget_with_recent_context_first(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    example = MessageExample("one", 1, "current", 0, ("old " * 100, "recent " * 100))
    dataset = ModerationDataset([example], use_context=True)
    batch = BertCollator(tokenizer, max_length=12)([dataset[0]])
    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    assert tokens[1] == "current"
    assert "recent" in tokens
    assert "old" not in tokens


def test_split_never_reuses_test_conversations_when_validation_empty():
    examples = [MessageExample(str(i), 0, "current", 0, ()) for i in range(4)]
    train, val, test = split_by_conversation(examples)
    assert not val
    assert {e.conversation_id for e in train}.isdisjoint(e.conversation_id for e in test)
    assert len(train) + len(test) == len(examples)


def test_context_window_zero_has_no_history(tmp_path):
    (tmp_path / "one.csv").write_text("text,label\ncurrent,0\nrecent,1\n")
    examples = load_conversations(tmp_path, context_window=0)
    assert all(not e.context for e in examples)
