# Hello Bert: Sentiment Analysis for Flying Food

![image.png](attachment:image.png)

End-to-end решение для анализа тональности отзывов сервиса доставки еды Flying Food с использованием классических методов машинного обучения и fine-tuning BERT.

## Оглавление

- [Описание проекта](#описание-проекта)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Результаты](#результаты)
- [Лицензия](#лицензия)

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

- BERT показывает наилучшие результаты благодаря контекстуальным эмбеддингам
- Классические модели быстрее в обучении и предсказании

## Контакты

Ваше Имя - [@ruslanprashchurovich](https://github.com/ruslanprashchurovich)

Ссылка на проект: [https://github.com/ruslanprashchurovich/hello-bert](https://github.com/ruslanprashchurovich/hello-bert)
