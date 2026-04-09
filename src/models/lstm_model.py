from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CONTEXT_WINDOW,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    SEQUENCE_DYNAMIC_COLUMNS,
    SEQUENCE_STATIC_COLUMNS,
)
from src.models.common import (
    TargetMinMaxNormalizer,
    GlobalTargetLogNormalizer,
    GlobalTargetMinMaxNormalizer,
    WindowedSequenceDataset,
    default_device,
    fit_category_maps,
    predictions_to_frame,
    save_json,
    set_seed,
    temporal_validation_split,
)

LOGGER = logging.getLogger(__name__)


def _sample_building_frame(
    frame: pd.DataFrame,
    max_buildings: int,
    random_seed: int,
) -> pd.DataFrame:
    building_ids = frame["building_id"].drop_duplicates()
    if len(building_ids) <= max_buildings:
        return frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    sampled_ids = (
        building_ids.sample(n=max_buildings, random_state=random_seed)
        .sort_values()
        .tolist()
    )
    sampled = frame[frame["building_id"].isin(sampled_ids)].copy()
    return sampled.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


@dataclass
class LSTMConfig:
    context_window: int = CONTEXT_WINDOW
    prediction_horizon: int = PREDICTION_HORIZON
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 10
    patience: int = 3
    train_stride: int = 24
    eval_stride: int = 24
    max_train_windows: int = 200_000
    max_eval_windows: int | None = None
    target_transform: str = "log1p"
    random_seed: int = RANDOM_SEED


class _LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        return self.head(outputs[:, -1, :])


class LSTMExperimentModel:
    def __init__(self, config: LSTMConfig | None = None, device: str | None = None) -> None:
        self.config = config or LSTMConfig()
        self.device = torch.device(default_device(device))
        self.model: _LSTMForecaster | None = None
        self.normalizer: GlobalTargetMinMaxNormalizer | GlobalTargetLogNormalizer | None = None
        self.category_maps: dict[str, dict[str, int]] | None = None
        self.best_val_loss: float | None = None

    def _build_normalizer(
        self,
        train_frame: pd.DataFrame,
    ) -> GlobalTargetMinMaxNormalizer | GlobalTargetLogNormalizer:
        if self.config.target_transform == "minmax":
            return GlobalTargetMinMaxNormalizer.fit(train_frame)
        if self.config.target_transform == "log1p":
            return GlobalTargetLogNormalizer.fit(train_frame)
        raise ValueError(f"Unsupported LSTM target_transform: {self.config.target_transform}")

    def tune_on_tsplit(
        self,
        train_frame: pd.DataFrame,
        max_buildings: int = 64,
    ) -> LSTMConfig:
        LOGGER.info(
            "LSTM tuning start rows=%s buildings=%s sampled_buildings=%s device=%s",
            len(train_frame),
            train_frame["building_id"].nunique(),
            max_buildings,
            self.device,
        )
        sampled = _sample_building_frame(
            train_frame,
            max_buildings=max_buildings,
            random_seed=self.config.random_seed,
        )
        base_config = asdict(self.config)
        tuning_budget = {
            "max_epochs": min(self.config.max_epochs, 4),
            "patience": min(self.config.patience, 2),
            "max_train_windows": min(self.config.max_train_windows, 60_000),
            "max_eval_windows": 10_000,
        }
        grid = [
            {},
            {"hidden_size": 192, "dropout": 0.1, "learning_rate": 1e-3},
            {"hidden_size": 128, "num_layers": 3, "dropout": 0.2, "learning_rate": 5e-4},
        ]

        best_config = LSTMConfig(**base_config)
        best_score = float("inf")

        for overrides in grid:
            candidate_config = LSTMConfig(**(base_config | overrides))
            tuning_config = LSTMConfig(**(asdict(candidate_config) | tuning_budget))
            candidate_model = LSTMExperimentModel(config=tuning_config, device=str(self.device))
            candidate_model.fit(sampled)
            score = (
                float(candidate_model.best_val_loss)
                if candidate_model.best_val_loss is not None
                else float("inf")
            )
            LOGGER.info("LSTM tuning params=%s best_val=%.6f", overrides or {"default": True}, score)
            if score < best_score:
                best_score = score
                best_config = candidate_config

        self.config = best_config
        LOGGER.info("LSTM tuning best_params=%s best_val=%.6f", asdict(self.config), best_score)
        return self.config

    def _build_datasets(
        self,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
    ) -> tuple[WindowedSequenceDataset, WindowedSequenceDataset]:
        self.category_maps = fit_category_maps(train_frame)
        self.normalizer = self._build_normalizer(train_frame)
        train_ds = WindowedSequenceDataset(
            train_frame,
            normalizer=self.normalizer,
            category_maps=self.category_maps,
            context_window=self.config.context_window,
            horizon=self.config.prediction_horizon,
            stride=self.config.train_stride,
            max_windows=self.config.max_train_windows,
            seed=self.config.random_seed,
        )
        valid_ds = WindowedSequenceDataset(
            valid_frame,
            normalizer=self.normalizer,
            category_maps=self.category_maps,
            context_window=self.config.context_window,
            horizon=self.config.prediction_horizon,
            stride=self.config.train_stride,
            max_windows=min(self.config.max_train_windows // 4, 50_000),
            seed=self.config.random_seed,
        )
        return train_ds, valid_ds

    def fit(self, train_frame: pd.DataFrame) -> "LSTMExperimentModel":
        set_seed(self.config.random_seed)
        inner_train, inner_valid = temporal_validation_split(train_frame)
        train_ds, valid_ds = self._build_datasets(inner_train, inner_valid)
        if len(train_ds) == 0:
            raise RuntimeError("No train windows were generated for LSTM.")
        LOGGER.info(
            "LSTM fit start device=%s train_windows=%s valid_windows=%s",
            self.device,
            len(train_ds),
            len(valid_ds),
        )

        input_dim = train_ds.inputs[0].shape[1]
        self.model = _LSTMForecaster(
            input_dim=input_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            horizon=self.config.prediction_horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.SmoothL1Loss()

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.config.batch_size, shuffle=False)

        best_state = None
        best_val = float("inf")
        patience_left = self.config.patience

        for epoch_idx in range(self.config.max_epochs):
            self.model.train()
            train_losses: list[float] = []
            for x_batch, y_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                if not torch.isfinite(x_batch).all() or not torch.isfinite(y_batch).all():
                    raise RuntimeError("Non-finite LSTM training batch detected.")
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(x_batch)
                loss = loss_fn(pred, y_batch)
                if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                    raise RuntimeError("Non-finite LSTM prediction or loss detected during training.")
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch, _ in valid_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    if not torch.isfinite(x_batch).all() or not torch.isfinite(y_batch).all():
                        raise RuntimeError("Non-finite LSTM validation batch detected.")
                    pred = self.model(x_batch)
                    if not torch.isfinite(pred).all():
                        raise RuntimeError("Non-finite LSTM validation prediction detected.")
                    val_losses.append(float(loss_fn(pred, y_batch).item()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            LOGGER.info(
                "LSTM epoch=%s/%s train_loss=%.6f val_loss=%.6f patience_left=%s",
                epoch_idx + 1,
                self.config.max_epochs,
                train_loss,
                val_loss,
                patience_left,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    LOGGER.info("LSTM early stopping triggered at epoch=%s", epoch_idx + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_loss = best_val
        LOGGER.info("LSTM fit finished best_val=%.6f", best_val)
        return self

    def predict_frame(
        self,
        frame: pd.DataFrame,
        split_name: str,
        fold_id: str | None = None,
    ) -> pd.DataFrame:
        if self.model is None or self.normalizer is None or self.category_maps is None:
            raise RuntimeError("Model must be fitted before prediction.")

        dataset = WindowedSequenceDataset(
            frame,
            normalizer=self.normalizer,
            category_maps=self.category_maps,
            context_window=self.config.context_window,
            horizon=self.config.prediction_horizon,
            stride=self.config.eval_stride,
            max_windows=self.config.max_eval_windows,
            seed=self.config.random_seed,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        sample_buildings: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            cursor = 0
            for x_batch, y_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                pred = self.model(x_batch).cpu().numpy()
                true = y_batch.numpy()
                if not np.isfinite(pred).all() or not np.isfinite(true).all():
                    raise RuntimeError("Non-finite LSTM prediction batch detected.")
                batch_size = pred.shape[0]
                batch_buildings = np.array(
                    [
                        dataset.sample_meta[idx]["building_id"]
                        for idx in range(cursor, cursor + batch_size)
                    ],
                    dtype=object,
                )
                cursor += batch_size
                preds.append(pred)
                truths.append(true)
                sample_buildings.append(batch_buildings)

        pred_norm = np.vstack(preds)
        true_norm = np.vstack(truths)
        building_ids = np.concatenate(sample_buildings)

        pred_denorm = self.normalizer.inverse_transform_rows(pred_norm, building_ids)
        true_denorm = self.normalizer.inverse_transform_rows(true_norm, building_ids)
        return predictions_to_frame(dataset, pred_denorm, true_denorm, split_name=split_name, model_name="lstm", fold_id=fold_id)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.normalizer is None or self.category_maps is None:
            raise RuntimeError("Model must be fitted before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        save_json(asdict(self.config), path.with_suffix(".config.json"))
        save_json(asdict(self.normalizer), path.with_suffix(".normalizer.json"))
        save_json(self.category_maps, path.with_suffix(".categories.json"))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "LSTMExperimentModel":
        path = Path(path)
        config_payload = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
        config = LSTMConfig(**(asdict(LSTMConfig()) | config_payload))
        loaded = cls(config=config, device=device)

        normalizer_payload = json.loads(path.with_suffix(".normalizer.json").read_text(encoding="utf-8"))
        if {"building_min", "building_max"}.issubset(normalizer_payload):
            loaded.normalizer = TargetMinMaxNormalizer(**normalizer_payload)
            loaded.config.target_transform = "minmax"
        elif {"global_min", "global_max"}.issubset(normalizer_payload):
            loaded.normalizer = GlobalTargetMinMaxNormalizer(**normalizer_payload)
            loaded.config.target_transform = "minmax"
        elif "offset" in normalizer_payload:
            loaded.normalizer = GlobalTargetLogNormalizer(**normalizer_payload)
            loaded.config.target_transform = "log1p"
        else:
            raise ValueError("Unsupported LSTM normalizer payload.")

        loaded.category_maps = json.loads(path.with_suffix(".categories.json").read_text(encoding="utf-8"))
        input_dim = 1 + len(SEQUENCE_DYNAMIC_COLUMNS) + len(SEQUENCE_STATIC_COLUMNS)
        loaded.model = _LSTMForecaster(
            input_dim=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            horizon=config.prediction_horizon,
        ).to(loaded.device)
        state_dict = torch.load(path, map_location=loaded.device)
        loaded.model.load_state_dict(state_dict)
        loaded.model.eval()
        return loaded
