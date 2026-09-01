from partial_monitoring import hf_local


def test_load_local_tokenizer_forces_local_files(monkeypatch):
    calls = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs
            return object()

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)
    hf_local.load_local_tokenizer("local/model")

    assert calls["model_path"] == "local/model"
    assert calls["kwargs"]["local_files_only"] is True


def test_load_local_model_forces_local_files(monkeypatch):
    calls = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs
            return object()

    monkeypatch.setattr("transformers.AutoModel", FakeAutoModel)
    hf_local.load_local_model("local/model")

    assert calls["model_path"] == "local/model"
    assert calls["kwargs"]["local_files_only"] is True
