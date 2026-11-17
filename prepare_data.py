"""Load -> clean -> dedupe -> stratified train/val/test split.

Uses the `clean()` function from clean_data.py unchanged. The cleaned CSV is
cached so you only pay the regex cost once (use --reclean in train.py to redo).
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from clean_data import clean  # your original cleaner

_EMOJI_TOKEN = re.compile(r":([a-z0-9_]+):")


def _despace_tokens(t: str) -> str:
    """':loudly_crying_face:' -> ':loudly crying face:' so WordPiece sees real words."""
    return _EMOJI_TOKEN.sub(lambda m: ":" + m.group(1).replace("_", " ") + ":", t)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as clean_data.main(), but returns the DataFrame."""
    n0 = len(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["statement", "status"]).copy()
    df["statement"] = df["statement"].astype(str).map(clean)
    df = df[df["statement"].str.contains(r"[A-Za-z]{2}") & (df["statement"].str.len() >= 3)]

    key = df["statement"].str.lower()
    # identical text with different labels -> label unreliable, drop all copies
    conflict = df.groupby(key)["status"].transform("nunique") > 1
    df = df[~conflict]
    df = df[~df["statement"].str.lower().duplicated()]
    df = df.reset_index(drop=True)
    print(f"[clean] {n0} rows in -> {len(df)} rows out")
    return df


def load_clean(src: str, cache: str, reclean: bool = False) -> pd.DataFrame:
    if Path(cache).exists() and not reclean:
        print(f"[clean] using cached {cache}")
        return pd.read_csv(cache)
    df = clean_dataframe(pd.read_csv(src))
    df.to_csv(cache, index=False)
    return df


def make_splits(df, seed=42, val_size=0.1, test_size=0.1, max_per_class=None):
    """Stratified split. Dedup already happened, so no exact text crosses splits."""
    df = df.copy()
    df["statement"] = df["statement"].map(_despace_tokens)

    if max_per_class:  # optional cap on big classes (mainly Normal) for speed/balance
        df = (df.groupby("status", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
                .reset_index(drop=True))

    labels = sorted(df["status"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    df["label"] = df["status"].map(label2id)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val["label"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True), label2id)
