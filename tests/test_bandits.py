import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from partial_monitoring.bandits import LinUCB, NeuralUCB, extract_bert_features_with_model
from partial_monitoring.data import MessageExample
from partial_monitoring.dataset import ModerationDataset


def encoder_fixture(tmp_path, *, vocab_size=7, max_positions=32):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]))
    tokenizer = BertTokenizer(str(vocab))
    model = BertModel(BertConfig(
        vocab_size=vocab_size, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=max_positions,
    ))
    dataset = ModerationDataset([
        MessageExample("one", 0, "hello", 0, ()),
        MessageExample("one", 1, "hello world", 1, ("hello",)),
    ])
    return tokenizer, model, dataset


def test_single_sequence_features_ignore_invalid_tokenizer_segment_ids(tmp_path):
    tokenizer, model, dataset = encoder_fixture(tmp_path)
    expected, labels, metadata = extract_bert_features_with_model(
        dataset, tokenizer, model, 2, 16, torch.device("cpu"),
    )

    class InvalidSegmentTokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, *args, **kwargs):
            encoded = tokenizer(*args, **kwargs)
            encoded["token_type_ids"] = torch.full_like(encoded["input_ids"], 17)
            return encoded

    actual, actual_labels, actual_metadata = extract_bert_features_with_model(
        dataset, InvalidSegmentTokenizer(), model, 2, 16, torch.device("cpu"),
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual_labels, labels)
    assert actual_metadata == metadata


def test_out_of_range_word_ids_are_rejected_before_cuda_transfer(tmp_path):
    tokenizer, model, dataset = encoder_fixture(tmp_path, vocab_size=6)
    # "world" has ID 6. No GPU is needed: the CPU check must fail before .to().
    with pytest.raises(ValueError, match=r"vocabulary mismatch.*0\.\.5"):
        extract_bert_features_with_model(
            dataset, tokenizer, model, 2, 16, torch.device("cuda"),
        )


def test_overlong_bert_sequence_is_rejected_before_cuda_transfer(tmp_path):
    tokenizer, model, dataset = encoder_fixture(tmp_path, max_positions=3)
    with pytest.raises(ValueError, match=r"4 tokens.*at most 3"):
        extract_bert_features_with_model(
            dataset, tokenizer, model, 2, 16, torch.device("cuda"),
        )


def test_batched_linucb_fit_matches_full_feedback_and_caches_inverse(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 4))
    y = rng.integers(0, 2, size=40)
    bandit = LinUCB(4, ridge=2)
    bandit.fit(x, y, false_negative_cost=25, positive_weight=3)
    reference = LinUCB(4, ridge=2)
    for feature, label in zip(x, y):
        reference.update_full_feedback(feature, int(label), .2, 1, 25, 3 if label else 1)
    np.testing.assert_allclose(bandit.a, reference.a)
    np.testing.assert_allclose(bandit.b, reference.b)
    scores, std = bandit.predict_removal_scores(x[:2])
    expected = x[:2] @ (np.linalg.solve(reference.a[0], reference.b[1] - reference.b[0]))
    np.testing.assert_allclose(scores, expected)
    assert np.all(std > 0)

    def unexpected_inverse(*args):
        raise AssertionError("Inference recomputed a cached inverse")

    monkeypatch.setattr(np.linalg, "inv", unexpected_inverse)
    bandit.decide(x[0])
    bandit.predict_removal_scores(x[:2])


def test_reward_policy_keeps_known_human_cost_and_alpha_is_conservative():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 3))
    bandit = LinUCB(3, alpha=0)
    bandit.fit(x, rng.integers(0, 2, size=30), class_weighting="none")
    unpenalized = bandit.decide(x[0])
    bandit.alpha = 2
    conservative = bandit.decide(x[0])
    assert conservative.scores[2] == -.2
    assert conservative.uncertainty[2] == 0
    assert all(conservative.scores[i] < unpenalized.scores[i] for i in (0, 1))


def test_neural_scores_are_batched_and_finite():
    x = np.random.default_rng(5).normal(size=(12, 3)).astype(np.float32)
    y = np.tile([0, 1], 6)
    bandit = NeuralUCB(3, epochs=1, batch_size=4, mc_samples=3, device_name="cpu")
    bandit.fit(x, y, class_weighting="none")
    scores, std = bandit.predict_removal_scores(x)
    assert scores.shape == std.shape == (12,)
    assert np.isfinite(scores).all()
    assert np.all(std >= 0)
    assert bandit.decide(x[0]).uncertainty[2] == 0
