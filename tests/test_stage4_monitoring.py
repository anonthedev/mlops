"""Unit tests for Stage 4 monitoring, drift PSI, group MAE, and promotion logic."""

import numpy as np
import pandas as pd
import pytest

from mp124.monitor import (
    clean_frame,
    compute_all_psi,
    compute_group_mae,
    compute_psi,
    should_promote,
    split_later_chronologically,
)


def test_clean_frame_deduplicates_and_cleans_sentinel():
    raw = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "biometric_failures": [9999, 9999, 4],
        "shop_area": [" Rural ", " Rural ", "URBAN."],
    })
    cleaned = clean_frame(raw)
    assert len(cleaned) == 2
    assert np.isnan(cleaned["biometric_failures"].iloc[0])
    assert cleaned["biometric_failures"].iloc[1] == 4
    assert cleaned["shop_area"].tolist() == ["rural", "urban"]


def test_compute_psi_detects_shift_and_zero_on_identical():
    ref = pd.Series(np.linspace(0, 100, 200))
    assert compute_psi(ref, ref, bins=10) == pytest.approx(0.0, abs=1e-9)
    shifted = pd.Series(np.linspace(500, 600, 200))
    assert compute_psi(ref, shifted, bins=10) > 1.0


def test_compute_all_psi_identifies_most_drifted_column():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "cards_attached": rng.normal(1000, 50, 200),
        "temp_c": rng.normal(30, 2, 200),
    })
    cur = pd.DataFrame({
        "cards_attached": rng.normal(2500, 50, 200),
        "temp_c": rng.normal(30, 2, 200),
    })
    scores = compute_all_psi(ref, cur, ["cards_attached", "temp_c"])
    assert max(scores, key=scores.get) == "cards_attached"


def test_compute_group_mae_flags_worst_block():
    df = pd.DataFrame({
        "block": ["Block North", "Block North", "Block South", "Block South", "Block East"],
        "grain_kg": [100.0, 110.0, 500.0, 600.0, 200.0],
        "_pred": [102.0, 108.0, 100.0, 150.0, 205.0],
    })
    mae_by_group = compute_group_mae(df, "grain_kg", "_pred")
    assert mae_by_group["Block North"] == pytest.approx(2.0)
    assert mae_by_group["Block East"] == pytest.approx(5.0)
    assert mae_by_group["Block South"] == pytest.approx(425.0)
    assert max(mae_by_group, key=mae_by_group.get) == "Block South"


def test_should_promote_10_percent_rule():
    assert should_promote(100.0, 90.0, 0.10) is True
    assert should_promote(100.0, 85.0, 0.10) is True
    assert should_promote(100.0, 90.01, 0.10) is False
    assert should_promote(167.756812, 120.956446, 0.10) is True


def test_split_later_chronologically_orders_by_date():
    df = pd.DataFrame({
        "date": ["2026-08-10", "2026-07-01", "2026-07-15", "2026-08-01", "2026-07-20",
                 "2026-08-20", "2026-07-05", "2026-08-15", "2026-07-25", "2026-08-25"],
        "val": range(10),
    })
    fit, check = split_later_chronologically(df, "date", 0.70)
    assert len(fit) == 7
    assert len(check) == 3
    assert fit["date"].max() <= check["date"].min()
