"""Классические модели против fine-tuned BERT на тесте из notebooks/bert_fine_tuning.ipynb.

Идея — классический аналог «предобучения» BERT: линейные модели на TF-IDF n-граммах обучаются
на остальных ~374k отзывах Amazon Fine Food (того же источника), а затем их выходы
комбинируются мета-логрегрессией, обученной на train-части проекта.

Защита от утечки: из внешних данных удаляются все отзывы, текст которых (после нормализации)
совпадает с любым из 15 691 отзывов проекта, и все отзывы того же пользователя
с тем же временем публикации (дубли одного отзыва на разные товары).
Внешние модели НЕ видят ни одной строки проекта, поэтому их предсказания на train проекта
честные и годятся для обучения мета-модели. Гиперпараметры мета-модели — по 5-fold CV на train.

Запуск (~35 минут на CPU, ~12 ГБ RAM):
    python scripts/prepare_data.py --reviews path/to/Reviews.csv
    python scripts/beat_bert.py --reviews path/to/Reviews.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import ID2LABEL, load_bert_split

N_PROJECT_ROWS = 15691
BERT_TEST_ACCURACY = 0.8085
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b|[!?]"


def clean(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r"<br\s*/?>", " ", regex=True)


def normalize(texts: pd.Series) -> pd.Series:
    return texts.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def load_external(reviews_path: str) -> pd.DataFrame:
    raw = pd.read_csv(reviews_path, usecols=["UserId", "Time", "Score", "Text"])
    project, rest = raw.iloc[:N_PROJECT_ROWS], raw.iloc[N_PROJECT_ROWS:]
    same_text = normalize(rest.Text).isin(set(normalize(project.Text)))
    same_user_time = pd.Series(list(zip(rest.UserId, rest.Time)), index=rest.index).isin(
        set(zip(project.UserId, project.Time))
    )
    ext = rest[~(same_text | same_user_time)].drop_duplicates("Text")
    print(
        f"Внешние отзывы: {len(rest)} -> удалено пересечений с проектом: "
        f"{(same_text | same_user_time).sum()} -> после дедупликации: {len(ext)}"
    )
    return ext


def log_proba(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="Путь к Reviews.csv (Amazon Fine Food)")
    args = parser.parse_args()
    t0 = time.time()

    train_df, val_df, test_df = load_bert_split()
    train_df = pd.concat([train_df, val_df], ignore_index=True)  # BERT использовал val для выбора эпохи
    y_train, y_test = train_df.label.values, test_df.label.values

    ext = load_external(args.reviews)
    stars = ext.Score.values
    y_ext = np.select([stars == 5, stars == 1], [0, 1], 2)  # та же разметка, что в проекте

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
        max_features=3_000_000,
        dtype=np.float32,
    )
    X_ext = vectorizer.fit_transform(clean(ext.Text))
    X_train = vectorizer.transform(clean(train_df.text))
    X_test = vectorizer.transform(clean(test_df.text))
    print(f"TF-IDF: {X_ext.shape[1]} признаков ({time.time() - t0:.0f} c)")

    # Базовые модели: обучены только на внешних данных, предсказывают train и test проекта.
    base_models = {
        "LR (3 класса)": (LogisticRegression(C=5, max_iter=1000), y_ext, "proba"),
        "LinearSVC (3 класса)": (LinearSVC(C=0.3, random_state=0), y_ext, "decision"),
        "LR (5 звёзд)": (LogisticRegression(C=5, max_iter=1000), stars, "proba"),
        "Ridge (звёзды)": (Ridge(alpha=1.0, solver="sparse_cg"), stars.astype(float), "regression"),
    }
    meta_train, meta_test = [], []
    for name, (model, target, kind) in base_models.items():
        t = time.time()
        model.fit(X_ext, target)
        if kind == "proba":
            a, b = log_proba(model.predict_proba(X_train)), log_proba(model.predict_proba(X_test))
        elif kind == "decision":
            a, b = model.decision_function(X_train), model.decision_function(X_test)
        else:
            a, b = model.predict(X_train)[:, None], model.predict(X_test)[:, None]
        meta_train.append(a)
        meta_test.append(b)
        msg = f"{name:22s} обучена за {time.time() - t:.0f} c"
        if a.shape[1] == 3:
            msg += f" | test accuracy: {accuracy_score(y_test, b.argmax(1)):.4f}"
        print(msg, flush=True)
    meta_train, meta_test = np.hstack(meta_train), np.hstack(meta_test)

    # Мета-модель: C выбирается только по кросс-валидации на train проекта.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    make_meta = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    cv_scores = {C: cross_val_score(make_meta(C), meta_train, y_train, cv=cv).mean() for C in (0.03, 0.1, 1.0)}
    best_C = max(cv_scores, key=cv_scores.get)
    meta = make_meta(best_C).fit(meta_train, y_train)
    pred = meta.predict(meta_test)
    acc = accuracy_score(y_test, pred)

    names = [ID2LABEL[i] for i in range(3)]
    print(f"\nМета-модель: C={best_C}, CV accuracy={cv_scores[best_C]:.4f}")
    print(classification_report(y_test, pred, target_names=names, digits=4))
    print("Матрица ошибок (строки — истина):")
    print(pd.DataFrame(confusion_matrix(y_test, pred), index=names, columns=names))
    se = np.sqrt(acc * (1 - acc) / len(y_test))
    print(
        f"\nTest accuracy: {acc:.4f} (95% ДИ {acc - 1.96 * se:.4f}–{acc + 1.96 * se:.4f}) "
        f"vs BERT {BERT_TEST_ACCURACY:.4f} -> {100 * (acc - BERT_TEST_ACCURACY):+.2f} п.п."
    )
    print(f"Всего: {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
