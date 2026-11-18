from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from partial_monitoring import hf_local


def test_load_local_tokenizer_forces_local_files(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    calls = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs
            return SimpleNamespace(pad_token="[PAD]", pad_token_id=0)

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)
    hf_local.load_local_tokenizer(str(tmp_path))

    assert calls["model_path"] == str(tmp_path.resolve())
    assert calls["kwargs"]["local_files_only"] is True


def save_tokenizer_without_pad_declaration(path, pad_id=0):
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]))
    tokenizer = BertTokenizer(str(vocab), pad_token=None)
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=7, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, pad_token_id=pad_id,
    )
    config.save_pretrained(path)
    return config


def test_missing_pad_declaration_is_restored_without_adding_tokens(tmp_path):
    config = save_tokenizer_without_pad_declaration(tmp_path)
    tokenizer = hf_local.load_local_tokenizer(str(tmp_path))

    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.pad_token_id == config.pad_token_id == 0
    assert len(tokenizer) == config.vocab_size == 7
    batch = tokenizer(["hello", "hello world"], padding=True, return_tensors="pt")
    assert batch["input_ids"][0, -1].item() == 0
    assert batch["attention_mask"][0, -1].item() == 0
    with torch.no_grad():
        output = BertModel(config).eval()(**batch)
    assert torch.isfinite(output.last_hidden_state).all()


@pytest.mark.parametrize("pad_id", [None, 1, 99])
def test_missing_pad_declaration_rejects_unknown_or_invalid_config_ids(tmp_path, pad_id):
    save_tokenizer_without_pad_declaration(tmp_path, pad_id=pad_id)
    with pytest.raises(RuntimeError, match="matching tokenizer files"):
        hf_local.load_local_tokenizer(str(tmp_path))


def test_load_local_model_forces_local_files(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    calls = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs
            return object()

    monkeypatch.setattr("transformers.AutoModel", FakeAutoModel)
    hf_local.load_local_model(str(tmp_path))

    assert calls["model_path"] == str(tmp_path.resolve())
    assert calls["kwargs"]["local_files_only"] is True


@pytest.mark.parametrize("loader", [hf_local.load_local_tokenizer, hf_local.load_local_model])
@pytest.mark.parametrize("case,error,match", [
    ("missing", FileNotFoundError, "missing or not visible"),
    ("file", NotADirectoryError, "not a file"),
    ("no_config", FileNotFoundError, "Missing local model configuration"),
])
def test_invalid_local_paths_are_rejected_before_transformers(tmp_path, monkeypatch, loader, case, error, match):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Transformers must not interpret an invalid local path as a Hub ID")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", unexpected_load)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", unexpected_load)
    path = tmp_path / "model"
    if case == "file":
        path.write_text("weights")
    elif case == "no_config":
        path.mkdir()
    with pytest.raises(error, match=match):
        loader(str(path))


def test_relative_local_path_is_resolved_from_job_working_directory(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert hf_local.describe_local_model_path("model") == str(model_dir.resolve())
