from __future__ import annotations

from pathlib import Path


def describe_local_model_path(model_path: str) -> str:
    """Resolve a local checkpoint before Transformers can interpret it as a Hub ID."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Local model path is missing or not visible in this job: {model_path!r}. "
            f"Resolved path: {path}. Working directory: {Path.cwd()}. "
            "Check --bert-model inside the SLURM job and pass the existing model directory."
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"--bert-model must point to a local model directory, not a file: {path}"
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing local model configuration: {path / 'config.json'}. "
            "Point --bert-model to the directory containing config.json, the tokenizer and the weights, "
            "rather than its parent directory."
        )
    return str(path)


def load_local_tokenizer(model_path: str):
    local_path = describe_local_model_path(model_path)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the tokenizer from local directory {local_path!r}: {exc}. "
            "On Calcul Canada, pass "
            "`--bert-model /path/to/local/model_dir` where the directory contains tokenizer files "
            "such as tokenizer.json or vocab.txt, plus config.json."
        ) from exc
    if tokenizer.pad_token_id is None:
        # Some local exports lose the special-token declaration while retaining
        # the original PAD vocabulary entry and its ID in the model config.
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(local_path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "The local tokenizer has no padding token and the local model config could not be read. "
                "Copy config.json and the tokenizer files saved with save_pretrained() into --bert-model."
            ) from exc
        pad_id = config.pad_token_id
        pad_token = tokenizer.convert_ids_to_tokens(pad_id) if pad_id is not None else None
        if (pad_token is None or tokenizer.get_vocab().get(pad_token) != pad_id
                or pad_id == tokenizer.unk_token_id):
            raise RuntimeError(
                "The local tokenizer has no padding token, and config.json does not identify an existing "
                "padding token in its vocabulary. Copy the matching tokenizer files, including "
                "special_tokens_map.json and tokenizer_config.json, into --bert-model."
            )
        tokenizer.pad_token = pad_token
    return tokenizer


def load_local_model(model_path: str):
    local_path = describe_local_model_path(model_path)
    try:
        from transformers import AutoModel

        return AutoModel.from_pretrained(local_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the BERT model from local directory {local_path!r}: {exc}. "
            "On Calcul Canada, pass "
            "`--bert-model /path/to/local/model_dir` where the directory contains config.json "
            "and model weights such as model.safetensors or pytorch_model.bin."
        ) from exc
