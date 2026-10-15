# Data Science Playlist Tutorials

**Learn by doing:** each lesson pairs an interactive [marimo](https://marimo.io) 📓 notebook with a YouTube 🎥 walkthrough—short, practical answers to classic data science questions.

## Tutorials

| # | Title | Marimo notebook | YouTube |
|---|--------|-----------------|--------|
| 1 | 🔢 One-Hot Encoding | [`marimo/onehot_encode.py`](marimo/onehot_encode.py) | [▶️ Watch](https://www.youtube.com/watch?v=7xGJXzxBZik&t=1017s) |
| 2 | 📦 Boxplot & Skewness | [`marimo/boxplot_and_skew.py`](marimo/boxplot_and_skew.py) | [▶️ Watch](https://www.youtube.com/watch?v=x8po9Qi1JO4&t=1187s) |

## 🚀 Run marimo notebook locally

From the project root (with [uv](https://docs.astral.sh/uv/) or your Python env):

```bash
uv sync
cd marimo
uv run marimo edit onehot_encode.py
```

Use `boxplot_and_skew.py` instead of `onehot_encode.py` for lesson 2. Notebooks load data from `marimo/data/ecommerce_dataset.csv` 📁.
