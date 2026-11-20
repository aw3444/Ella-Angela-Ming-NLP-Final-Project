import numpy as np
import pytest

from src.metrics import compute_metrics


def test_compute_metrics():
    logits = np.array([
        [
            [5.0, 1.0],  # predicts 0
            [1.0, 5.0],  # predicts 1
            [1.0, 5.0],  # ignored
            [5.0, 1.0],  # predicts 0
        ]
    ])

    labels = np.array([
        [0, 1, -100, 1]
    ])

    metrics = compute_metrics((logits, labels))

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)