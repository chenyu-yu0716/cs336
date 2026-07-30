# CS336 — Language Modeling From Scratch

Stanford CS336 coursework monorepo.

## Contents

| Directory | Description |
|-----------|-------------|
| [`assignment1/`](assignment1/) | **2026** Assignment 1 starter ([upstream](https://github.com/stanford-cs336/assignment1-basics), v26.x) |
| [`assignment1-2025/`](assignment1-2025/) | Archived **2025** Assignment 1 work (reference only; do not mix with 2026) |

Work in `assignment1/` for the current course. Treat `assignment1-2025/` as read-only archive.

## Setup (2026 Assignment 1)

```bash
cd assignment1
uv run pytest   # initially fails with NotImplementedError until you fill in adapters
```

### Download data

Datasets and model checkpoints are **not** tracked by git:

```bash
mkdir -p assignment1/data && cd assignment1/data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```
