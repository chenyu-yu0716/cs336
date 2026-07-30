# CS336 — Language Modeling From Scratch

Stanford CS336 coursework monorepo.

## Contents

| Directory | Description |
|-----------|-------------|
| [`assignment1/`](assignment1/) | Assignment 1: Basics ([upstream](https://github.com/stanford-cs336/assignment1-basics)) |

## Setup

Each assignment manages its own environment with `uv`. See the README inside the assignment folder.

### Download data (Assignment 1)

Datasets are **not** tracked by git. From the assignment directory:

```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

Model checkpoints (`*.pt`, `*.safetensors`, `checkpoints/`, etc.) are also gitignored.
