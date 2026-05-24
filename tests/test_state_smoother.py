"""Tests for the 3-of-5 hysteresis StateSmoother."""
from __future__ import annotations

from mac_app.inference.state_smoother import StateSmoother


def test_bootstrap_uses_first_pred() -> None:
    s = StateSmoother(window_size=5, threshold=3)
    out = s.update("occupied", 0.9, 0.5, ts="t0")
    assert out.stable == "occupied"
    assert out.raw == "occupied"


def test_holds_state_until_threshold_disagrees() -> None:
    s = StateSmoother(window_size=5, threshold=3)
    for ts in range(5):
        s.update("vacant", 0.95, 0.0, ts=f"t{ts}")
    assert s.stable == "vacant"
    # Two occupied predictions are not enough.
    s.update("occupied", 0.8, 0.5, ts="t5")
    s.update("occupied", 0.8, 0.5, ts="t6")
    assert s.stable == "vacant"
    # A third pushes the count of "occupied" in the last 5 to >= 3.
    out = s.update("occupied", 0.8, 0.5, ts="t7")
    assert out.stable == "occupied"
    assert out.last_changed_at == "t7"


def test_does_not_flip_on_single_outlier() -> None:
    s = StateSmoother(window_size=5, threshold=3)
    for ts in range(5):
        s.update("occupied", 0.9, 0.3, ts=f"t{ts}")
    # One stray vacant pred shouldn't change anything.
    out = s.update("vacant", 0.6, 0.0, ts="t5")
    assert out.stable == "occupied"


def test_threshold_must_be_strict() -> None:
    # 2 of 5 isn't enough; 3 of 5 is.
    s = StateSmoother(window_size=5, threshold=3)
    for ts in range(5):
        s.update("vacant", 0.95, 0.0, ts=f"t{ts}")
    s.update("occupied", 0.8, 0.5, ts="t5")
    s.update("occupied", 0.8, 0.5, ts="t6")
    assert s.stable == "vacant"


def test_threshold_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        StateSmoother(window_size=3, threshold=10)
