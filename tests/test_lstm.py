from __future__ import annotations

import inspect

import numpy as np
import pytest

from lstm_bitcoin.config import LSTMSettings
from lstm_bitcoin.models.lstm import TorchLSTM


def settings() -> LSTMSettings:
    return LSTMSettings(
        hidden_size=3,
        layers=1,
        epochs=3,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        patience=2,
        gradient_clip=1.0,
    )


def arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 40)
    features = np.stack(
        [
            np.column_stack((time[index : index + 4], time[index : index + 4] ** 2))
            for index in range(30)
        ]
    ).astype(np.float64)
    targets = (features[:, -1, 0] * 0.02).astype(np.float64)
    return features[:22], targets[:22], features[22:26], targets[22:26]


def test_predict_signature_is_feature_only() -> None:
    parameters = list(inspect.signature(TorchLSTM.predict).parameters)
    assert parameters == ["self", "features"]


def test_unfitted_lstm_rejects_prediction() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TorchLSTM(settings(), seed=7).predict(np.ones((2, 4, 2)))


@pytest.mark.parametrize(
    ("train_x", "train_y", "validation_x", "validation_y", "message"),
    [
        (np.ones((3, 2)), np.ones(3), np.ones((2, 3, 1)), np.ones(2), "three-dimensional"),
        (np.ones((3, 3, 1)), np.ones((3, 1)), np.ones((2, 3, 1)), np.ones(2), "one-dimensional"),
        (np.ones((3, 3, 1)), np.ones(2), np.ones((2, 3, 1)), np.ones(2), "align"),
        (np.ones((1, 3, 1)), np.ones(1), np.ones((2, 3, 1)), np.ones(2), "must not be empty"),
        (np.ones((3, 3, 1)), np.ones(3), np.ones((0, 3, 1)), np.ones(0), "must not be empty"),
        (np.full((3, 3, 1), np.nan), np.ones(3), np.ones((2, 3, 1)), np.ones(2), "finite"),
    ],
)
def test_lstm_fit_validation(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TorchLSTM(settings(), seed=7).fit(train_x, train_y, validation_x, validation_y)


@pytest.mark.lstm
def test_torch_lstm_is_deterministic_and_predicts_features_only() -> None:
    pytest.importorskip("torch")
    train_x, train_y, validation_x, validation_y = arrays()

    first = TorchLSTM(settings(), seed=7).fit(train_x, train_y, validation_x, validation_y)
    second = TorchLSTM(settings(), seed=7).fit(train_x, train_y, validation_x, validation_y)
    first_prediction = first.predict(validation_x)
    second_prediction = second.predict(validation_x)

    np.testing.assert_array_equal(first_prediction, second_prediction)
    assert first.summary == second.summary
    assert first.summary is not None
    assert 1 <= first.summary.best_epoch <= first.summary.epochs_ran <= settings().epochs
    assert np.isfinite(first.summary.best_validation_loss)
    assert np.isfinite(first.summary.final_training_loss)

    with pytest.raises(ValueError, match="three-dimensional"):
        first.predict(np.ones((2, 4)))


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[lstm\]"):
        TorchLSTM._torch()
