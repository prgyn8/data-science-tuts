# Data Science Playlist Tutorials

A curated set of essential data science concepts, each in its own runnable [marimo](https://marimo.io) 📓 notebook under `marimo/`. Use the table below to open a topic, experiment in the notebook, and optionally follow the matching YouTube 🎥 walkthrough.

## Tutorials

| # | Title | Marimo notebook | YouTube |
|---|--------|-----------------|--------|
| 1 | 🔢 One-Hot Encoding | [`marimo/onehot_encode.py`](marimo/onehot_encode.py) | [▶️ Watch](https://www.youtube.com/watch?v=7xGJXzxBZik&t=1017s) |
| 2 | 📦 Boxplot & Skewness | [`marimo/boxplot_and_skew.py`](marimo/boxplot_and_skew.py) | [▶️ Watch](https://www.youtube.com/watch?v=x8po9Qi1JO4&t=1187s) |
| 3 | 📐 Principal Component Analysis (PCA) | [`marimo/PCA.py`](marimo/PCA.py) | [▶️ Watch](#) |
| 4 | 📈 Logistic Regression | [`marimo/logistic_regression.py`](marimo/logistic_regression.py) | [▶️ Watch](#) |

## 🚀 Run marimo notebook locally

From the project root (with [uv](https://docs.astral.sh/uv/) or your Python env):

```bash
uv sync
cd marimo
uv run marimo edit onehot_encode.py
```

Use `boxplot_and_skew.py` instead of `onehot_encode.py` for lesson 2. Notebooks load data from `marimo/data/ecommerce_dataset.csv` 📁.
