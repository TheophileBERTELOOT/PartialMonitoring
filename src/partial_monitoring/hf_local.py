from __future__ import annotations

from pathlib import Path


def describe_local_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)
    return model_path


def load_local_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(describe_local_model_path(model_path), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the tokenizer from local files. On Calcul Canada, pass "
            "`--bert-model /path/to/local/model_dir` where the directory contains tokenizer files "
            "such as tokenizer.json or vocab.txt, plus config.json."
        ) from exc


def load_local_model(model_path: str):
    try:
        from transformers import AutoModel

        return AutoModel.from_pretrained(describe_local_model_path(model_path), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the BERT model from local files. On Calcul Canada, pass "
            "`--bert-model /path/to/local/model_dir` where the directory contains config.json "
            "and model weights such as model.safetensors or pytorch_model.bin."
        ) from exc
