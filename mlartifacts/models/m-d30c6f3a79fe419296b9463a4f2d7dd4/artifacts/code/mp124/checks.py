"""The seven data-quality checks from the Student Handbook, section 7.

Every function takes the raw train.csv as a DataFrame ("as given": duplicates
are NOT removed first, as the handbook asks) and returns a dict whose keys
match the matching block of reports/findings.yaml, plus an "evidence" entry
that explains in words what the code saw. Run them all with:

    python -m mp124.find_issues
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def normalise_text(value: object) -> object:
    """Lower-case, trim spaces and drop a trailing dot: ' Tribal ' -> 'tribal',
    'tri.' -> 'tri'. Missing values are left alone."""
    if pd.isna(value):
        return value
    return str(value).strip().lower().rstrip(".")


def map_to_levels(value: object, levels: list[str]) -> object:
    """Map a messy spelling onto one of the clean levels.

    First normalise it; if it is then an exact clean level, keep it. Otherwise
    accept it only if it is a prefix of exactly ONE clean level (so 'rur'
    becomes 'rural'). Anything ambiguous or unknown is returned unchanged so a
    human can look at it.
    """
    norm = normalise_text(value)
    if pd.isna(norm) or norm in levels:
        return norm
    matches = [lvl for lvl in levels if lvl.startswith(norm)]
    return matches[0] if len(matches) == 1 else norm


def drop_sentinels(s: pd.Series, sentinel_candidates: list[float]) -> pd.Series:
    """The column without the special 'typed instead of blank' numbers, so they
    do not distort the other checks (means, runs ...)."""
    return s[~s.isin(sentinel_candidates)]


# --------------------------------------------------------------------------
# 1. leaking column
# --------------------------------------------------------------------------
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str,
                         id_col: str, corr_threshold: float) -> dict:
    """A leaking column is filled in only AFTER the day is over, so:
    (a) it is missing from holdout_features.csv (the file we must predict), or
    (b) it agrees almost exactly with the target (|correlation| very high)."""
    missing_in_holdout = [c for c in train.columns
                          if c not in holdout.columns and c != target]
    numeric = train.select_dtypes("number").drop(columns=[target], errors="ignore")
    corr = numeric.corrwith(train[target]).abs().sort_values(ascending=False)
    too_close = corr[corr >= corr_threshold].index.tolist()
    suspects = list(dict.fromkeys(missing_in_holdout + too_close))
    return {
        "present": bool(suspects),
        "column": suspects[0] if suspects else None,
        "evidence": (f"columns in train but not in holdout (besides the target): "
                     f"{missing_in_holdout}; highest |corr| with {target}: "
                     f"{corr.round(3).to_dict()}"),
    }


# --------------------------------------------------------------------------
# 2. unit mix-up
# --------------------------------------------------------------------------
def check_unit_mixup(train: pd.DataFrame, numeric_cols: list[str], group_col: str,
                     sentinel_candidates: list[float], unit_factors: list[float],
                     ratio_threshold: float) -> dict:
    """For each numeric column compare the mean for each group value (block)
    with the median of the other groups' means. A group that is about 10, 12,
    60, 100 or 1000 times bigger/smaller recorded the column in another unit."""
    table = {}
    for col in numeric_cols:
        clean = train.loc[~train[col].isin(sentinel_candidates), [col, group_col]]
        means = clean.groupby(group_col)[col].mean()
        for grp, m in means.items():
            others = means.drop(grp).median()
            if others == 0 or m == 0:
                continue
            ratio = m / others
            table[(col, grp)] = ratio
            big = max(ratio, 1 / ratio)
            if big >= ratio_threshold:
                # the exact unit factor closest to the observed ratio
                factor = min(unit_factors, key=lambda f: abs(np.log(big / f)))
                return {
                    "present": True, "column": col, "group_value": grp,
                    "factor": factor,
                    "evidence": f"{col}: mean in {grp} is {ratio:.2f}x the other blocks",
                }
    worst = max(table.items(), key=lambda kv: max(kv[1], 1 / kv[1]))
    return {
        "present": False, "column": None, "group_value": None, "factor": None,
        "evidence": (f"largest block-vs-others mean ratio is {worst[1]:.3f} "
                     f"({worst[0][0]} in {worst[0][1]}); every column looks the "
                     f"same in all blocks"),
    }


# --------------------------------------------------------------------------
# 3. sentinel value
# --------------------------------------------------------------------------
def check_sentinel_value(train: pd.DataFrame, numeric_cols: list[str],
                         sentinel_candidates: list[float]) -> dict:
    """A sentinel is a special number (-1, -999, 9999 ...) typed instead of a
    blank. We flag a candidate that appears 2+ times AND lies outside the
    range of all the other values in that column (so it is impossible)."""
    found = []
    for col in numeric_cols:
        counts = train[col].value_counts()
        for val in sentinel_candidates:
            n = int(counts.get(val, 0))
            if n < 2:
                continue
            rest = train.loc[train[col] != val, col]
            if val > rest.max() or val < rest.min():
                found.append((col, val, n, rest.min(), rest.max()))
    if not found:
        return {"present": False, "column": None, "value": None, "count": None,
                "evidence": "no candidate sentinel number found outside a column's normal range"}
    col, val, n, lo, hi = max(found, key=lambda t: t[2])
    return {
        "present": True, "column": col, "value": int(val) if float(val).is_integer() else val,
        "count": n,
        "evidence": f"{col} = {val} on {n} rows; all other values lie in [{lo}, {hi}]",
    }


# --------------------------------------------------------------------------
# 4. duplicate rows
# --------------------------------------------------------------------------
def check_duplicate_rows(train: pd.DataFrame, id_col: str) -> dict:
    """df.duplicated() marks every copy after the first, so its sum is the
    number of EXTRA rows we would delete."""
    extra = int(train.duplicated().sum())
    extra_no_id = int(train.drop(columns=[id_col]).duplicated().sum())
    return {
        "present": extra > 0,
        "count": extra if extra > 0 else None,
        "evidence": (f"{extra} fully identical extra rows; {extra_no_id} if "
                     f"{id_col} is ignored; {int(train[id_col].duplicated().sum())} "
                     f"repeated {id_col}s"),
    }


# --------------------------------------------------------------------------
# 5. stuck sensor
# --------------------------------------------------------------------------
def check_stuck_sensor(train: pd.DataFrame, numeric_cols: list[str], date_col: str,
                       sentinel_candidates: list[float], min_days: int) -> dict:
    """Sort by date; look for the longest run of consecutive rows where a column
    keeps the same value. A run that covers many days = a stuck device."""
    df = train.sort_values(date_col, kind="stable").reset_index(drop=True)
    best = None  # (n_days, col, value, start, end, n_rows)
    for col in numeric_cols:
        run_id = (df[col] != df[col].shift()).cumsum()
        runs = df.groupby(run_id).agg(value=(col, "first"), n_rows=(col, "size"),
                                      start=(date_col, "min"), end=(date_col, "max"),
                                      n_days=(date_col, "nunique"))
        runs = runs[~runs["value"].isin(sentinel_candidates)]
        top = runs.sort_values(["n_days", "n_rows"]).iloc[-1]
        cand = (int(top.n_days), col, top.value, top.start, top.end, int(top.n_rows))
        if best is None or cand[0] > best[0]:
            best = cand
    n_days, col, val, start, end, n_rows = best
    if n_days >= min_days:
        return {"present": True, "column": col, "start_date": str(start),
                "end_date": str(end),
                "evidence": f"{col} stuck at {val} for {n_rows} rows over {n_days} days"}
    return {"present": False, "column": None, "start_date": None, "end_date": None,
            "evidence": (f"longest same-value run is {n_rows} rows over {n_days} day(s) "
                         f"({col} = {val}, {start}); far below {min_days} days")}


# --------------------------------------------------------------------------
# 6. target typos (extra zero)
# --------------------------------------------------------------------------
def check_target_typos(train: pd.DataFrame, target: str, gap_ratio: float) -> dict:
    """Sort the target. Typed with an extra zero, a few values sit ~10x above
    the rest with an empty gap below them. We find the biggest jump between
    neighbouring sorted values in the upper half; if the next value is at least
    `gap_ratio` times the one below it, everything above the jump is a typo."""
    v = np.sort(train[target].dropna().to_numpy())
    upper = v[len(v) // 2:]
    jumps = upper[1:] / upper[:-1]
    i = int(np.argmax(jumps))
    if jumps[i] >= gap_ratio:
        n = int((v > upper[i]).sum())
        return {"present": True, "count": n,
                "evidence": f"{n} values above {upper[i]:.1f}, next value {upper[i+1]:.1f} ({jumps[i]:.1f}x jump)"}
    return {"present": False, "count": None,
            "evidence": (f"largest jump between neighbouring sorted {target} values "
                         f"in the upper half is {jumps[i]:.3f}x "
                         f"({upper[i]} -> {upper[i+1]}); max {v[-1]} is only "
                         f"{v[-1] / np.median(v):.2f}x the median")}


# --------------------------------------------------------------------------
# 7. spelling variants
# --------------------------------------------------------------------------
def check_spelling_variants(train: pd.DataFrame, reference: pd.DataFrame,
                            cat_cols: list[str]) -> dict:
    """value_counts() on every text column. later.csv has the clean spellings,
    so we use its levels as the 'true' categories and map every messy
    spelling onto them."""
    for col in cat_cols:
        raw_levels = train[col].dropna().unique()
        true_levels = sorted(reference[col].dropna().map(normalise_text).unique())
        cleaned = train[col].map(lambda x: map_to_levels(x, true_levels))
        unmapped = sorted(set(cleaned.dropna()) - set(true_levels))
        if len(raw_levels) > len(true_levels):
            return {
                "present": True, "column": col,
                "true_levels": int(cleaned.nunique()),
                "evidence": (f"{col} has {len(raw_levels)} raw spellings "
                             f"{sorted(map(repr, raw_levels))}; after cleaning: "
                             f"{cleaned.value_counts().to_dict()}; unmapped: {unmapped}"),
            }
    return {"present": False, "column": None, "true_levels": None,
            "evidence": "every text column already uses only the clean spellings"}


# --------------------------------------------------------------------------
def run_all_checks(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame,
                   params: dict) -> dict:
    """Run the seven checks in the order of findings.yaml."""
    c, k = params["columns"], params["checks"]
    return {
        "leaking_column": check_leaking_column(train, holdout, c["target"], c["id"],
                                               k["leak_corr_threshold"]),
        "unit_mixup": check_unit_mixup(train, c["numeric"], c["group"],
                                       k["sentinel_candidates"], k["unit_factors"],
                                       k["unit_ratio_threshold"]),
        "sentinel_value": check_sentinel_value(train, c["numeric"], k["sentinel_candidates"]),
        "duplicate_rows": check_duplicate_rows(train, c["id"]),
        "stuck_sensor": check_stuck_sensor(train, c["numeric"], c["date"],
                                           k["sentinel_candidates"], k["stuck_min_days"]),
        "target_typos": check_target_typos(train, c["target"], k["typo_gap_ratio"]),
        "spelling_variants": check_spelling_variants(train, later, c["categorical"]),
    }
