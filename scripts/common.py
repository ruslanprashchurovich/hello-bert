"""Загрузка данных и сплит, побайтно повторяющий notebooks/bert_fine_tuning.ipynb."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "food_reviews.csv"
# Порядок как в ноутбуке BERT (label2id по df["label"].unique()) — важно для стратификации.
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_bert_split():
    """Возвращает (train_df, val_df, test_df) ровно как в ноутбуке BERT.

    Колонка score (1-5 звёзд) используется только как вспомогательная цель при обучении.
    """
    df = pd.read_csv(DATA, sep=",", usecols=[2, 3, 4], header=0, names=["score", "label", "text"])
    # Дубликаты считаются по (label, text), как в ноутбуке; score просто «едет» рядом.
    df = df.drop_duplicates(subset=["label", "text"], keep="first", ignore_index=True)
    df["label"] = df["label"].map(LABEL2ID)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
