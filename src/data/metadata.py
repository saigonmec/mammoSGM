"""metadata.csv handling: label encoding, patient-grouped validation split, sampler.

Expected columns: `link` (image path relative to data_folder), `split` ("train"/"test",
optionally "val"), and the target column (default "cancer"). A patient/case id column
(`patient_id`, `PatientID`, `subject_id`, `case_id`, or as a fallback `image_id`/`id`) is used
to keep all images of one patient on the same side of every split.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

LABEL_COL = "label"
_STRONG_GROUP_COLS = ("patient_id", "patientid", "patient", "subject_id", "case_id")
_WEAK_GROUP_COLS = ("image_id", "id")  # usually "<patient>_<L|R>_<view>"
_VIEW_SUFFIX = re.compile(r"_(?:LCC|RCC|LMLO|RMLO|MLO|CC|L|R)(?=_|$).*$")
_VAL_NAMES = {"val", "valid", "validation"}


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def read_metadata(data_folder: str) -> pd.DataFrame:
    path = os.path.join(data_folder, "metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.csv not found in {data_folder}")
    df = pd.read_csv(path)
    for col in ("link", "split"):
        if col not in df.columns:
            raise KeyError(f"metadata.csv is missing required column '{col}'")
    return df


def _native(v):
    return v.item() if hasattr(v, "item") else v


def load_labeled_frame(data_folder: str, target_column: str = "cancer"):
    """All rows with an integer `label` column (0..C-1, by sorted target value).
    Returns (df, class_names) where class_names[i] is the original value of class i."""
    df = read_metadata(data_folder)
    if target_column not in df.columns:
        raise KeyError(f"target column '{target_column}' not in metadata.csv ({list(df.columns)})")
    df = df.drop_duplicates(subset=["link"])
    df = df[df[target_column].notna()].copy()
    col = df[target_column]
    if pd.api.types.is_numeric_dtype(col) and np.all(np.mod(col.unique(), 1) == 0):
        col = col.astype(int)
    class_names = [_native(v) for v in sorted(col.unique())]
    to_idx = {v: i for i, v in enumerate(class_names)}
    df[LABEL_COL] = col.map(to_idx).astype(int)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df.loc[df["split"].isin(_VAL_NAMES), "split"] = "val"
    return df.reset_index(drop=True), class_names


def find_group_column(df: pd.DataFrame, override: str | None = None):
    """-> (column name | None, is_weak). Weak ids get view/side suffixes stripped."""
    lower = {c.lower(): c for c in df.columns}
    if override:
        if override not in df.columns:
            raise KeyError(f"group column '{override}' not in metadata")
        return override, False
    for name in _STRONG_GROUP_COLS:
        if name in lower:
            return lower[name], False
    for name in _WEAK_GROUP_COLS:
        if name in lower:
            return lower[name], True
    return None, False


def _groups(df: pd.DataFrame, col, weak: bool) -> pd.Series:
    if col is None:
        return pd.Series(df["link"].values, index=df.index)  # image-level fallback
    s = df[col].astype(str)
    return s.map(lambda v: _VIEW_SUFFIX.sub("", v)) if weak else s


def _grouped_stratified_split(df: pd.DataFrame, groups: pd.Series, frac: float, seed: int):
    """Hold out ~`frac` of the *groups*, stratified by 'has class c' (max label per group)."""
    rng = np.random.default_rng(seed)
    gl = pd.DataFrame({"g": groups.values, "y": df[LABEL_COL].values}).groupby("g")["y"].max()
    val_groups: set = set()
    for _, members in gl.groupby(gl):
        names = np.array(sorted(members.index))
        rng.shuffle(names)
        k = int(round(len(names) * frac))
        if len(names) >= 2:
            k = min(max(k, 1), len(names) - 1)
        else:
            k = 0
        val_groups.update(names[:k].tolist())
    is_val = groups.isin(val_groups).values
    return df[~is_val], df[is_val]


def _dist_table(name: str, df: pd.DataFrame, class_names) -> str:
    counts = df[LABEL_COL].value_counts().reindex(range(len(class_names)), fill_value=0)
    cells = "  ".join(f"{class_names[i]}:{int(counts[i])}" for i in range(len(class_names)))
    return f"  {name:<5} n={len(df):<6} {cells}"


def load_metadata(
    data_folder: str,
    target_column: str = "cancer",
    val_ratio: float = 0.1,
    seed: int = 42,
    group_column: str | None = None,
    make_val: bool = True,
    verbose: bool = True,
):
    """-> (train_df, val_df | None, test_df, class_names).

    * If metadata has a "val" split it is used as-is.
    * Otherwise `val_ratio` of the *train patients* are held out (stratified).
    * val_ratio == 0 reuses the test set for model selection (loud warning: optimistic!).
    * Train/val/test are checked for patient overlap.
    """
    df, class_names = load_labeled_frame(data_folder, target_column)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    val_df = df[df["split"] == "val"]
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"metadata needs 'train' and 'test' rows, got splits: {df['split'].value_counts().to_dict()}")

    gcol, weak = find_group_column(df, group_column)
    if gcol is None:
        _warn("no patient/case id column found: splits are image-level, so images of one patient "
              "may leak between train/val/test. Add a 'patient_id' column.")
    elif weak:
        _warn(f"using '{gcol}' (view/side suffix stripped) as patient id; add 'patient_id' to be exact.")

    if not make_val:
        val_df = None
    elif len(val_df) == 0:
        if val_ratio and val_ratio > 0:
            groups = _groups(train_df, gcol, weak)
            train_df, val_df = _grouped_stratified_split(train_df, groups, val_ratio, seed)
            if verbose:
                print(f"Created validation split from train: {len(val_df)} images "
                      f"({val_ratio:.0%} of train patients, seed={seed}).")
        else:
            _warn("val_ratio=0 and no 'val' split: using the TEST set for model selection / early "
                  "stopping. Reported test numbers will be optimistically biased.")
            val_df = test_df

    if gcol is not None:
        sets = {"train": train_df, "test": test_df}
        if val_df is not None and val_df is not test_df:
            sets["val"] = val_df
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = set(_groups(sets[names[i]], gcol, weak))
                b = set(_groups(sets[names[j]], gcol, weak))
                if a & b:
                    _warn(f"PATIENT LEAKAGE: {len(a & b)} patient ids appear in both "
                          f"'{names[i]}' and '{names[j]}'.")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)

    if verbose:
        print(f"Target '{target_column}' -> classes {dict(enumerate(class_names))}")
        print(_dist_table("train", train_df, class_names))
        if val_df is not None:
            print(_dist_table("val", val_df, class_names) + ("  (== test)" if val_df is test_df else ""))
        print(_dist_table("test", test_df, class_names))
    return train_df, val_df, test_df, class_names


def class_counts(labels, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)


def balanced_class_weights(labels, num_classes: int) -> torch.Tensor:
    """N / (C * n_c), the usual 'balanced' class weights."""
    counts = np.maximum(class_counts(labels, num_classes), 1)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def get_weighted_sampler(labels, num_classes: int, generator: torch.Generator | None = None):
    """Draw len(labels) samples/epoch with probability ~ 1 / class frequency."""
    labels = np.asarray(labels, dtype=int)
    w = 1.0 / np.maximum(class_counts(labels, num_classes), 1)[labels]
    return WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True, generator=generator
    )
