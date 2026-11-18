from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BertCollator, ModerationDataset
from .hf_local import load_local_model
from .models import NeuralRewardModel
from .policy import ACTION_NAMES, INDEX_TO_ACTION, action_rewards


@dataclass(frozen=True)
class BanditDecision:
    action: str
    action_index: int
    scores: list[float]
    uncertainty: list[float]


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _validate_encoder_input_ids(input_ids: torch.Tensor, model: nn.Module, tokenizer) -> None:
    """Check indices on CPU before an invalid embedding lookup poisons CUDA."""
    embedding_count = model.get_input_embeddings().num_embeddings
    smallest, largest = int(input_ids.min()), int(input_ids.max())
    if smallest < 0 or largest >= embedding_count:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: input_ids range [{smallest}, {largest}], "
            f"but the model accepts IDs 0..{embedding_count - 1}. "
            f"Tokenizer: {getattr(tokenizer, 'name_or_path', '<unknown>')!r}. "
            "Copy the weights and matching tokenizer from the same checkpoint into --bert-model; "
            "do not add vocabulary entries without compatible model embeddings."
        )
    # BERT uses a fixed position embedding table. Other architectures may use
    # relative positions or position offsets, so do not infer their limits here.
    if getattr(model.config, "model_type", None) == "bert":
        position_count = model.embeddings.position_embeddings.num_embeddings
        if input_ids.shape[1] > position_count:
            raise ValueError(
                f"Encoded sequence has {input_ids.shape[1]} tokens, but this BERT model supports "
                f"at most {position_count}. Set --max-length to {position_count} or less."
            )


@torch.no_grad()
def extract_bert_features(
    dataset: ModerationDataset,
    tokenizer,
    bert_model_name: str,
    batch_size: int = 32,
    max_length: int = 256,
    device_name: str = "auto",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    device = get_device(device_name)
    model = load_local_model(bert_model_name).to(device)
    return extract_bert_features_with_model(dataset, tokenizer, model, batch_size, max_length, device)


@torch.no_grad()
def extract_bert_features_with_model(
    dataset: ModerationDataset,
    tokenizer,
    model: nn.Module,
    batch_size: int,
    max_length: int,
    device: torch.device,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=BertCollator(tokenizer, max_length))

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for batch in tqdm(loader, desc="BERT embeddings", leave=False):
        _validate_encoder_input_ids(batch["input_ids"], model, tokenizer)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # The collator encodes one string per example, including any context.
        # Let BERT use segment zero throughout rather than forwarding segment
        # IDs from a potentially malformed local tokenizer postprocessor.
        outputs = model(**kwargs)
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        features.append(pooled.detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
        for conversation_id, message_index in zip(batch["conversation_ids"], batch["message_indices"]):
            metadata.append({"conversation_id": conversation_id, "message_index": message_index})

    return np.vstack(features), np.concatenate(labels), metadata


class LinUCB:
    def __init__(self, feature_dim: int, alpha: float = 1.0, ridge: float = 1.0) -> None:
        self.alpha = alpha
        self.a = np.stack([np.eye(feature_dim, dtype=np.float64) * ridge for _ in ACTION_NAMES])
        self.b = np.zeros((len(ACTION_NAMES), feature_dim), dtype=np.float64)
        self._inv_a = None
        self._theta = None
        self.reward_noise = np.ones(len(ACTION_NAMES))
        self.margin_noise = 1.0
        self.human_cost = 0.2

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        human_cost: float = 0.2,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        class_weighting: str = "balanced",
        positive_weight: float | None = None,
    ) -> dict[str, float]:
        weights = sample_weights(labels, class_weighting, positive_weight)
        self.human_cost = human_cost
        rewards = np.asarray([
            action_rewards(int(y), human_cost, false_positive_cost, false_negative_cost) for y in labels
        ])
        # Full feedback shares one Gram matrix across actions; use batched BLAS.
        for start in range(0, len(features), 4096):
            x = features[start:start + 4096].astype(np.float64)
            w = weights[start:start + 4096]
            gram = x.T @ (w[:, None] * x)
            self.a += gram[None, :, :]
            self.b += (x.T @ (w[:, None] * rewards[start:start + 4096])).T
        self._inv_a = None
        self._theta = None
        self._parameters()
        residual_sums = np.zeros(len(ACTION_NAMES))
        margin_residual_sum = 0.0
        for start in range(0, len(features), 4096):
            residual = rewards[start:start + 4096] - features[start:start + 4096] @ self._theta.T
            w = weights[start:start + 4096]
            residual_sums += (w[:, None] * residual ** 2).sum(axis=0)
            margin_residual_sum += float((w * (residual[:, 1] - residual[:, 0]) ** 2).sum())
        self.reward_noise = np.sqrt(residual_sums / weights.sum())
        self.reward_noise[2] = 0.0
        self.margin_noise = float(np.sqrt(margin_residual_sum / weights.sum()))
        return training_stats(labels, weights)

    def update_full_feedback(
        self,
        x: np.ndarray,
        label: int,
        human_cost: float,
        false_positive_cost: float,
        false_negative_cost: float,
        sample_weight: float,
    ) -> None:
        rewards = action_rewards(label, human_cost, false_positive_cost, false_negative_cost)
        self.human_cost = human_cost
        outer = np.outer(x, x)
        for action_index, reward in enumerate(rewards):
            self.a[action_index] += sample_weight * outer
            self.b[action_index] += sample_weight * reward * x
        if self._inv_a is not None:
            projected = self._inv_a @ x
            self._inv_a -= sample_weight * np.outer(projected, projected) / (1 + sample_weight * x @ projected)
            self._theta = self.b @ self._inv_a
        else:
            self._theta = None

    def _parameters(self) -> None:
        if self._theta is None:
            self._inv_a = np.linalg.inv(self.a[0])
            self._theta = np.linalg.solve(self.a[0], self.b.T).T

    def predict_removal_scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._parameters()
        scores = features @ (self._theta[1] - self._theta[0])
        std = np.empty(len(features))
        for start in range(0, len(features), 4096):
            x = features[start:start + 4096]
            std[start:start + 4096] = np.sqrt(np.maximum(0, np.einsum("ij,ij->i", x @ self._inv_a, x))) * self.margin_noise
        return scores, std

    def decide(self, x: np.ndarray) -> BanditDecision:
        self._parameters()
        width = float(np.sqrt(max(0.0, x @ self._inv_a @ x)))
        uncertainty = (width * self.reward_noise).tolist()
        means = self._theta @ x
        means[2] = -self.human_cost
        scores = (means - self.alpha * np.asarray(uncertainty)).tolist()
        action_index = int(np.argmax(scores))
        return BanditDecision(INDEX_TO_ACTION[action_index], action_index, scores, uncertainty)


class NeuralUCB:
    def __init__(
        self,
        feature_dim: int,
        alpha: float = 1.0,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        mc_samples: int = 20,
        device_name: str = "auto",
    ) -> None:
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.device = get_device(device_name)
        self.model = NeuralRewardModel(feature_dim, hidden_dim, dropout).to(self.device)
        self.human_cost = 0.2

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        human_cost: float = 0.2,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        class_weighting: str = "balanced",
        positive_weight: float | None = None,
    ) -> dict[str, float]:
        weights = sample_weights(labels, class_weighting, positive_weight)
        self.human_cost = human_cost
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(
            [action_rewards(int(label), human_cost, false_positive_cost, false_negative_cost) for label in labels],
            dtype=torch.float32,
        )
        w = torch.tensor(weights, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y, w)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        last_loss = 0.0

        for _ in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y, batch_w in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_x)
                per_example_loss = (predictions - batch_y).pow(2).mean(dim=1)
                loss = (per_example_loss * batch_w).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            last_loss = total_loss / max(1, len(loader))
        return training_stats(labels, weights) | {"train_loss": last_loss}

    @torch.no_grad()
    def decide(self, x: np.ndarray) -> BanditDecision:
        feature = torch.tensor(x[None, :], dtype=torch.float32, device=self.device)
        samples: list[np.ndarray] = []
        self.model.train()
        for _ in range(self.mc_samples):
            samples.append(self.model(feature).detach().cpu().numpy()[0])
        stacked = np.stack(samples)
        means = stacked.mean(axis=0)
        uncertainty = stacked.std(axis=0)
        means[2] = -self.human_cost
        uncertainty[2] = 0.0
        scores = means - self.alpha * uncertainty
        action_index = int(np.argmax(scores))
        return BanditDecision(
            INDEX_TO_ACTION[action_index],
            action_index,
            scores.astype(float).tolist(),
            uncertainty.astype(float).tolist(),
        )

    @torch.no_grad()
    def predict_removal_scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores, std = [], []
        self.model.train()
        for start in range(0, len(features), self.batch_size):
            x = torch.as_tensor(features[start:start + self.batch_size], dtype=torch.float32, device=self.device)
            samples = torch.stack([self.model(x) for _ in range(self.mc_samples)])
            margins = samples[:, :, 1] - samples[:, :, 0]
            scores.append(margins.mean(dim=0).cpu().numpy())
            std.append(margins.std(dim=0, unbiased=False).cpu().numpy())
        return np.concatenate(scores), np.concatenate(std)


def sample_weights(labels: np.ndarray, class_weighting: str = "balanced", positive_weight: float | None = None) -> np.ndarray:
    labels = labels.astype(np.int64)
    weights = np.ones_like(labels, dtype=np.float64)
    if positive_weight is not None:
        weights[labels == 1] = positive_weight
        return weights
    if class_weighting == "none":
        return weights
    if class_weighting != "balanced":
        raise ValueError(f"Unknown class_weighting: {class_weighting}")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    total = len(labels)
    if positives:
        weights[labels == 1] = total / (2 * positives)
    if negatives:
        weights[labels == 0] = total / (2 * negatives)
    return weights


def training_stats(labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    return {
        "train_examples": float(len(labels)),
        "train_positives": float(positives),
        "train_negatives": float(negatives),
        "train_positive_rate": positives / max(1, len(labels)),
        "mean_positive_weight": float(weights[labels == 1].mean()) if positives else 0.0,
        "mean_negative_weight": float(weights[labels == 0].mean()) if negatives else 0.0,
    }
