"""Small deterministic CPU LSTM with a feature-only inference boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import LSTMSettings
from .base import SequenceStandardizer, TargetStandardizer


class MissingLSTMDependency(RuntimeError):
    """Raised when the optional PyTorch dependency is unavailable."""


@dataclass(frozen=True)
class TrainingSummary:
    """Bounded training diagnostics saved with each fold and seed."""

    best_epoch: int
    epochs_ran: int
    best_validation_loss: float
    final_training_loss: float


class TorchLSTM:
    """One-step return forecaster backed by a compact PyTorch LSTM."""

    def __init__(self, settings: LSTMSettings, *, seed: int) -> None:
        self.settings = settings
        self.seed = seed
        self._network: Any | None = None
        self._feature_scaler: SequenceStandardizer | None = None
        self._target_scaler: TargetStandardizer | None = None
        self.summary: TrainingSummary | None = None

    @staticmethod
    def _torch() -> tuple[Any, Any, Any]:
        try:
            import torch
            from torch import nn
            from torch.utils import data
        except ImportError as exc:  # pragma: no cover - exercised without optional dependency
            raise MissingLSTMDependency(
                'PyTorch is required for the LSTM; install with "pip install -e .[lstm]"'
            ) from exc
        return torch, nn, data

    def fit(
        self,
        train_features: npt.NDArray[np.float64],
        train_targets: npt.NDArray[np.float64],
        validation_features: npt.NDArray[np.float64],
        validation_targets: npt.NDArray[np.float64],
    ) -> TorchLSTM:
        """Fit using only the earlier train block and stop on the later validation block."""

        if train_features.ndim != 3 or validation_features.ndim != 3:
            raise ValueError("LSTM features must be three-dimensional")
        if train_targets.ndim != 1 or validation_targets.ndim != 1:
            raise ValueError("LSTM targets must be one-dimensional")
        if len(train_features) != len(train_targets) or len(validation_features) != len(
            validation_targets
        ):
            raise ValueError("LSTM feature and target arrays must align")
        if len(train_features) < 2 or len(validation_features) < 1:
            raise ValueError("LSTM train and validation arrays must not be empty")
        arrays = (train_features, train_targets, validation_features, validation_targets)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("LSTM train and validation arrays must be finite")

        torch_module, nn_module, data_module = self._torch()
        torch = torch_module
        nn = nn_module
        torch_data = data_module
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        np.random.seed(self.seed)

        self._feature_scaler = SequenceStandardizer.fit(train_features)
        self._target_scaler = TargetStandardizer.fit(train_targets)
        train_x = torch.as_tensor(
            self._feature_scaler.transform(train_features), dtype=torch.float32
        )
        train_y = torch.as_tensor(
            self._target_scaler.transform(train_targets)[:, None], dtype=torch.float32
        )
        validation_x = torch.as_tensor(
            self._feature_scaler.transform(validation_features), dtype=torch.float32
        )
        validation_y = torch.as_tensor(
            self._target_scaler.transform(validation_targets)[:, None], dtype=torch.float32
        )

        # The optional dependency is imported lazily, so the base class is known only at runtime.
        class Network(nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, input_size: int, hidden_size: int, layers: int) -> None:
                super().__init__()
                self.recurrent = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=layers,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, features: Any) -> Any:
                sequence, _ = self.recurrent(features)
                return self.head(sequence[:, -1, :])

        network = Network(
            input_size=train_features.shape[-1],
            hidden_size=self.settings.hidden_size,
            layers=self.settings.layers,
        )
        loss_function = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed)
        dataset = torch_data.TensorDataset(train_x, train_y)
        loader = torch_data.DataLoader(
            dataset,
            batch_size=min(self.settings.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, object] | None = None
        stale_epochs = 0
        final_training_loss = float("nan")
        epochs_ran = 0
        for epoch in range(1, self.settings.epochs + 1):
            network.train()
            weighted_loss = 0.0
            seen = 0
            for batch_features, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = network(batch_features)
                loss = loss_function(prediction, batch_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.settings.gradient_clip)
                optimizer.step()
                batch_size = len(batch_features)
                weighted_loss += float(loss.detach()) * batch_size
                seen += batch_size
            final_training_loss = weighted_loss / seen

            network.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(network(validation_x), validation_y))
            epochs_ran = epoch
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.settings.patience:
                    break

        if best_state is None or not np.isfinite(best_loss):
            raise RuntimeError("LSTM training did not produce a finite validation loss")
        network.load_state_dict(best_state)
        network.eval()
        self._network = network
        self.summary = TrainingSummary(
            best_epoch=best_epoch,
            epochs_ran=epochs_ran,
            best_validation_loss=best_loss,
            final_training_loss=final_training_loss,
        )
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict from past-only features; targets are intentionally not accepted."""

        if self._network is None or self._feature_scaler is None or self._target_scaler is None:
            raise RuntimeError("TorchLSTM must be fitted before prediction")
        if features.ndim != 3:
            raise ValueError("LSTM features must be three-dimensional")
        torch_module, _, _ = self._torch()
        torch = torch_module
        tensor = torch.as_tensor(self._feature_scaler.transform(features), dtype=torch.float32)
        self._network.eval()
        with torch.no_grad():
            normalized = self._network(tensor).cpu().numpy().reshape(-1)
        return self._target_scaler.inverse(normalized.astype(np.float64))
