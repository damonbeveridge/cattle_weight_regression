# Getting Started

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
uv sync
```

## Preparing data

Place the raw `.tar.gz` archive in `data/raw/`, then run:

```bash
uv run python scripts/prepare_data.py --archive data/raw/images.tar.gz
```

This extracts the images and validates the labels CSV.

## Running MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Open <http://localhost:5000> to browse experiment runs. The database file `mlruns.db`
is created automatically on first use (i.e. after your first training run).
