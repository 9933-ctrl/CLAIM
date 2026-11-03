
## Contents

- `claim/config.py`: 16 categories, 33 BIO labels, paper hyperparameters.
- `claim/data.py`, `prepare_data.py`: JSONL schema, character-aligned fast-tokenizer mapping, poem-level splits.
- `claim/model.py`, `claim/crf.py`, `claim/spans.py`: BERT-wwm-Chinese -> BiLSTM (256 per direction) -> linear-chain CRF.
- `claim/kg.py`, `train_kg.py`: TransE 256-dimensional embeddings and top-5 cultural triple lookup.
- `claim/fusion.py`: auxiliary MLP category head combining 512D span features and 256D KG entity features.
- `claim/rag.py`: structured RAG prompt, optional Hugging Face causal generator, transparent **heuristic** evidence check.
- `claim/metrics.py`, `evaluate.py`: exact-match recognition macro-F1 and generation metrics.
- `train.py`, `predict.py`: training and inference.
- `tests/`: sanity checks.

## Install

Python 3.10+ is recommended. BERT checkpoints require an internet connection at first download.

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Quick start

```bash
python train_kg.py --triples data/triples.jsonl --output outputs/transe.pt --epochs 10
python train.py --train data/train.jsonl --dev data/dev.jsonl --kg data/triples.jsonl --kg-embeddings outputs/transe.pt --epochs 2 --batch-size 2
python evaluate.py --test data/test.jsonl
python predict.py --dynasty Tang --kg-embeddings outputs/transe.pt
```

No generator argument = clearly labeled, extractive KG evidence baseline, **not a 72B language model**. To use a generative model:

```bash
python predict.py --kg-embeddings outputs/transe.pt --generator Qwen/Qwen2.5-1.5B-Instruct
```

Change to an appropriate available Qwen-72B-Instruct checkpoint if you have the storage/VRAM. The article calls its decoding "greedy" while also listing T=0.3/top-p=0.9; greedy mode (`do_sample=False`) makes sampling parameters inapplicable. This implementation uses deterministic greedy generation.

## Real dataset schema

Each line of the annotated input file:

```json
{"id":"poem_001","text":"床前明月光","dynasty":"Tang","spans":[{"start":2,"end":4,"label":"Moon"}]}
```

Spans use 0-based Unicode character indexing and an exclusive `end`; overlapping annotations are invalid for single-label BIO. The 16 category names are in `claim/config.py`. To split a real corpus:

```bash
python prepare_data.py --input data/my_annotated_poems.jsonl --output-dir data
```

The simple provided splitter is *poem-level random*, not a fully stratified multilabel splitter. For publication experiments, perform poem-grouped iterative stratification, avoid near-duplicate poems across splits, and keep augmented variants only in training.

KG JSONL: `{"head":"明月","relation":"SYMBOLISES","tail":"思乡"}`.

