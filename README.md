# CLAIM: Python modular reproduction

Implementation based on the supplied manuscript **Knowledge-Graph-Augmented Large Language Model Framework for Classical Chinese Literary Imagery Recognition and Cultural Meaning Generation**.

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

## Quick start (toy sample only)

```bash
python train_kg.py --triples data/triples.jsonl --output outputs/transe.pt --epochs 10
python train.py --train data/train.jsonl --dev data/dev.jsonl --kg data/triples.jsonl --kg-embeddings outputs/transe.pt --epochs 2 --batch-size 2
python evaluate.py --test data/test.jsonl
python predict.py --text '举头望明月，低头思故乡。' --dynasty Tang --kg-embeddings outputs/transe.pt
```

No generator argument = clearly labeled, extractive KG evidence baseline, **not a 72B language model**. To use a generative model:

```bash
python predict.py --text '举头望明月，低头思故乡。' --kg-embeddings outputs/transe.pt --generator Qwen/Qwen2.5-1.5B-Instruct
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

## Methodology and reproducibility notes

1. The actual **SinoImagery-12K poem-level BIO annotations**, **48,000 approved KG triples**, **2,400 archaic vocabulary additions**, experts' reference explanations, and the original checkpoint are **not supplied** in the manuscript. The supplied `data/` files are explicitly illustrative and the same examples are intentionally copied into train/dev/test solely to smoke-test the commands. **Do not report metrics on these toy splits.** Replace the data and KG and regenerate independent train/dev/test partitions before any experimental evaluation.
2. The paper describes a joint CRF/generator loss with `lambda_gen=0.3` but does not provide the Qwen conditioning/integration details. This code implements the **CRF and an auxiliary KG-fusion cross-entropy training objective** (weighted 0.3), while the generator is used separately at inference time. This is an approximation, **not** end-to-end joint 72B training or an exact implementation of the published loss.
3. The MLP fusion is an auxiliary gold-span classifier during training and can rerank detected span labels at inference, provided KG embeddings are available. The KG cannot alter CRF boundaries directly. This resolves an architectural ambiguity in the paper and should be described accurately in experiments.
4. A full KG fact verification model and 0.7 claim-level precision/recall threshold are not specified. `evidence_coverage` checks literal references to KG tails; it is **not** a semantic verifier and cannot establish hallucination-rate reductions.
5. The article's category-distribution table has totals that do not agree with its 12,400 annotations claim. The sample code does not fabricate the missing instances or reproduce the quoted 91.0% macro-F1 / BLEU-4 38.2 results.
6. BERT and optional generator weights are downloaded from model providers when the code is run. This project ships **source code only**, not pretrained model weights.
7. The current implementation does **not** reproduce the five specialized augmentation procedures, the extra-token vocabulary modifications, or all stated baseline/ablation experiments. Add these after obtaining the genuine annotation/augmentation resources.

## Compute

Qwen-72B full inference requires substantial accelerator memory. For a functional small demonstration use Qwen2.5-1.5B (still a checkpoint download and substantial RAM). For CPU-level checks, run the tests, TransE training, and inspect the sample RAG retrieval without loading a text model.
