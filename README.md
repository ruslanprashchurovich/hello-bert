# Hello Bert: Sentiment Analysis for Flying Food

![BERT Architecture](images/bert.png)

End-to-end решение для анализа тональности отзывов сервиса доставки еды Flying Food с использованием классических методов машинного обучения и fine-tuning BERT.

## Оглавление

- [Описание проекта](#описание-проекта)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Результаты](#результаты)

## Описание проекта

Проект решает задачу анализа тональности отзывов о сервисе доставки еды Flying Food. Реализованы два подхода:

1. Классические ML модели (Naive Bayes, Logistic Regression, Gradient Boosting) с использованием CountVectorizer/TF-IDF
2. Fine-tuning предобученной модели BERT для достижения максимальной точности

### Задача

- Вход: Текстовые отзывы клиентов
- Выход: Предсказание тональности (положительный/отрицательный/нейтральный)
- Метрика: Accuracy

## Структура проекта

```

hello-bert/
├── data/
├── notebooks/
└── configs/ # Конфигурационные файлы
```

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/ruslanprashchurovich/hello-bert.git
cd hello-bert
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Установка пакета

```bash
pip install -e .
```

## Результаты

### Сравнение моделей

| Модель                | Accuracy  |
| --------------------- | --------- |
| Naive Bayes           | 73.8%     |
| Logistic Regression   | 77.2%     |
| Gradient Boosting     | 70.1%     |
| **BERT (fine-tuned)** | **80.3%** |

### Выводы

- BERT показывает наилучшие результаты среди моделей, обученных только на данных проекта
- Классические модели быстрее в обучении и предсказании

### Классика против BERT: реванш

Подробный разбор с комментариями к каждому шагу — в [`notebooks/classical_beats_bert.ipynb`](notebooks/classical_beats_bert.ipynb). Скрипты в [`scripts/`](scripts/) обучают классические модели и проверяют их на **том же тестовом сплите**, что и BERT в `bert_fine_tuning.ipynb`: 3019 отзывов, те же дедупликация и `random_state`. Это подтверждается распределением классов 1904/278/837, совпадающим с матрицей ошибок BERT.

| Модель (тест BERT, 3019 отзывов)                                            | Accuracy   |
| --------------------------------------------------------------------------- | ---------- |
| BERT fine-tuned (Trainer API)                                               | 80.85%     |
| Стекинг 8 классических моделей, только данные проекта                       | 79.86%     |
| LinearSVC на TF-IDF 1–3-грамм + внешние отзывы                              | 82.11%     |
| **Стекинг: LR/SVM/Ridge на TF-IDF + внешние отзывы, мета-LR (выбор по CV)** | **82.41%** |

Как это работает. BERT выигрывает за счёт предобучения на огромном внешнем корпусе. Классический аналог: обучить линейные модели на TF-IDF n-граммах по остальным ~374k отзывам [Amazon Fine Food Reviews](https://huggingface.co/datasets/jhan21/amazon-food-reviews-dataset) (данные проекта — первые 15 691 строк этого датасета). Затем мета-логрегрессия комбинирует их выходы на train-части проекта. Помогают и вспомогательные цели по звёздам (5-классовая LR и регрессия Ridge): они различают 2★/3★/4★ внутри класса «neutral».

Защита от утечки. Из внешних данных удалены все отзывы, текст которых совпадает с любым отзывом проекта, а также отзывы того же пользователя с тем же временем публикации (всего 27 195). Внешние модели не видят ни одной строки проекта.

Честная оговорка. Без внешних данных классический стекинг даёт 79.86% и до BERT не дотягивает. Внешний корпус размечен звёздами, то есть это больше размеченных данных, чем видел fine-tuned BERT.

```bash
python scripts/prepare_data.py --reviews path/to/Reviews.csv   # собирает data/food_reviews.csv
python scripts/beat_bert.py --reviews path/to/Reviews.csv      # ~30 мин на CPU
```

## Контакты

Ваше Имя - [@ruslanprashchurovich](https://github.com/ruslanprashchurovich)

Ссылка на проект: [https://github.com/ruslanprashchurovich/hello-bert](https://github.com/ruslanprashchurovich/hello-bert)
