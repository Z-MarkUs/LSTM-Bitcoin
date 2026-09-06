from __future__ import annotations

import numpy as np
import pytest

from lstm_bitcoin.models.base import SequenceStandardizer, TargetStandardizer
from lstm_bitcoin.models.baselines import HistoricalMean, RidgeReturn, ZeroReturn


def training_arrays(samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(samples * 3 * 2, dtype=np.float64).reshape(samples, 3, 2)
    values[:, :, 1] = 5.0
    targets = 0.25 * values[:, -1, 0] - 2.0
    return values, targets


def test_sequence_standardizer_uses_training_moments_and_constant_fallback() -> None:
    features, _ = training_arrays()
    scaler = SequenceStandardizer.fit(features)
    transformed = scaler.transform(features)

    np.testing.assert_allclose(np.mean(transformed[:, :, 0]), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.std(transformed[:, :, 0]), 1.0, atol=1e-15)
    np.testing.assert_allclose(transformed[:, :, 1], 0.0)
    assert scaler.scale[1] == 1.0

    test = features[:2].copy()
    first = scaler.transform(test)
    appended = np.concatenate((test, np.full_like(test, 1e12)))
    np.testing.assert_allclose(scaler.transform(appended)[:2], first)


@pytest.mark.parametrize("features", [np.ones((2, 3)), np.ones((0, 3, 2))])
def test_sequence_standardizer_fit_rejects_bad_shapes(features: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty three-dimensional"):
        SequenceStandardizer.fit(features)


def test_sequence_standardizer_rejects_nonfinite_and_transform_mismatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        SequenceStandardizer.fit(np.full((2, 3, 1), np.nan))

    scaler = SequenceStandardizer.fit(np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="do not match"):
        scaler.transform(np.ones((2, 3, 2)))


def test_target_standardizer_round_trip_and_constant_fallback() -> None:
    targets = np.asarray([1.0, 2.0, 3.0])
    scaler = TargetStandardizer.fit(targets)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(targets)), targets)
    constant = TargetStandardizer.fit(np.ones(3))
    assert constant.scale == 1.0
    np.testing.assert_allclose(constant.transform(np.ones(3)), 0.0)


@pytest.mark.parametrize("targets", [np.ones((2, 1)), np.asarray([]), np.asarray([1.0, np.inf])])
def test_target_standardizer_rejects_bad_targets(targets: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"one-dimensional|finite"):
        TargetStandardizer.fit(targets)


def test_zero_and_historical_mean_baselines() -> None:
    features, targets = training_arrays()
    test = features[:4] * 100

    zero = ZeroReturn().fit(features, targets)
    np.testing.assert_array_equal(zero.predict(test), np.zeros(4))
    mean = HistoricalMean()
    with pytest.raises(RuntimeError, match="fitted"):
        mean.predict(test)
    mean.fit(features, targets)
    np.testing.assert_allclose(mean.predict(test), np.mean(targets))

    with pytest.raises(ValueError, match="three-dimensional"):
        zero.predict(np.ones((2, 3)))
    with pytest.raises(ValueError, match="three-dimensional"):
        mean.predict(np.ones((2, 3)))


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        (np.ones((2, 3)), np.ones(2)),
        (np.ones((2, 3, 1)), np.ones((2, 1))),
        (np.ones((2, 3, 1)), np.ones(3)),
        (np.ones((0, 3, 1)), np.ones(0)),
        (np.full((2, 3, 1), np.nan), np.ones(2)),
        (np.ones((2, 3, 1)), np.asarray([1.0, np.inf])),
    ],
)
def test_baseline_fit_validation(features: np.ndarray, targets: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"incompatible|non-empty and finite"):
        ZeroReturn().fit(features, targets)


def test_ridge_fits_linear_signal_and_scaler_is_train_only() -> None:
    features, targets = training_arrays(samples=50)
    model = RidgeReturn(alpha=1e-8).fit(features, targets)
    predictions = model.predict(features)
    np.testing.assert_allclose(predictions, targets, atol=1e-7)

    test = features[:2].copy()
    prediction = model.predict(test)
    appended = np.concatenate((test, np.full_like(test, 1e12)))
    np.testing.assert_allclose(model.predict(appended)[:2], prediction)


def test_ridge_validation_and_unfitted_prediction() -> None:
    with pytest.raises(ValueError, match="alpha"):
        RidgeReturn(alpha=-1)
    model = RidgeReturn()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.ones((2, 3, 1)))


def test_ridge_alpha_zero_handles_rank_deficient_design() -> None:
    features = np.ones((8, 3, 2), dtype=np.float64)
    targets = np.arange(8, dtype=np.float64)
    model = RidgeReturn(alpha=0).fit(features, targets)
    assert np.isfinite(model.predict(features)).all()
