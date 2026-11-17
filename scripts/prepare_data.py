"""Собирает data/food_reviews.csv из полного Amazon Fine Food Reviews (Hugging Face).

Источник: https://huggingface.co/datasets/jhan21/amazon-food-reviews-dataset (Reviews.csv, 568k отзывов).
Датасет проекта = первые 15 691 строк: score 5 -> positive, 1 -> negative, 2-4 -> neutral.

Запуск:
    python scripts/prepare_data.py --reviews path/to/Reviews.csv
"""

import argparse
from pathlib import Path

import pandas as pd

N_PROJECT_ROWS = 15691
ROOT = Path(__file__).resolve().parents[1]


def score_to_sentiment(score: int) -> str:
    if score == 5:
        return "positive"
    if score == 1:
        return "negative"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="Путь к Reviews.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.reviews)
    project = raw.iloc[:N_PROJECT_ROWS]
    out = pd.DataFrame(
        {
            "id": project["Id"],
            "time": pd.to_datetime(project["Time"], unit="s").dt.strftime("%d/%m/%y %H:%M"),
            "score": project["Score"],
            "sentiment": project["Score"].map(score_to_sentiment),
            "text": project["Text"],
        }
    )
    out.to_csv(ROOT / "data" / "food_reviews.csv", index=False)
    print(f"food_reviews.csv: {len(out)} строк")


if __name__ == "__main__":
    main()
