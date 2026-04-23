# Architecture

## System purpose

Binary classification of code snippets as human-written or AI-generated, for SemEval2026 Task 13A. The system implements a 7-run experiment roadmap progressing from a TF-IDF baseline to a calibrated ensemble.

## System diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ configs/                                                            │
│  global.yaml ◄── run{N}_*.yaml (inherits_from)                     │
└────────┬────────────────────────────────────────────────────────────┘
         │ --config path
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ src/train.py                                                        │
│  1. load_config() — YAML with shallow inheritance                   │
│  2. seed_everything(42)                                             │
│  3. Dispatch by model.type / model.architecture:                    │
│       "tfidf_lr"       → runners/run_tfidf.py                      │
│       "cls_classifier" → runners/run_transformer.py                 │
│       other            → ValueError                                 │
└────────┬──────────────────────────┬─────────────────────────────────┘
         │                          │
         ▼                          ▼
┌────────────────────┐  ┌───────────────────────────────────────────┐
│ run_tfidf.py       │  │ run_transformer.py                        │
│ (Run 1)            │  │ (Runs 2/3/4)                              │
│                    │  │                                           │
│ TfidfVectorizer    │  │ CodeClassifier                            │
│ + LogisticRegr.    │  │  ├─ AutoModel (CodeBERT or UniXcoder)     │
│                    │  │  ├─ Dropout                               │
│ sklearn pipeline   │  │  └─ Linear(hidden, 2)                    │
│ No GPU needed      │  │                                           │
│                    │  │ Selects dataset by config:                 │
│                    │  │  ├─ CodeDataset (truncation, 512 tokens)  │
│                    │  │  └─ HeadTailCodeDataset (256 head + tail) │
│                    │  │                                           │
│                    │  │ Training: AdamW + linear LR + AMP         │
└────────┬───────────┘  └────────────┬──────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ runs/<run_name>/                                                    │
│   config.yaml    — copy of config used                              │
│   train.log      — training logs                                    │
│   metrics.json   — validation metrics                               │
│   oof.csv        — validation predictions (prob, pred, true)        │
│   submission.csv — Kaggle submission (ID, label)                    │
│   best_model.pt  — checkpoint (transformer runs only)               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data flow

```
data/raw/train.parquet ──┐
data/raw/validation.parquet ──┤── preprocess.py ──► DataFrames
data/raw/test.parquet ───┘     (normalize line endings,
                                compute code_length, num_lines)
                                        │
                    ┌───────────────────┤
                    ▼                   ▼
            TfidfVectorizer      Tokenizer (HuggingFace)
                    │                   │
                    ▼                   ▼
            LogisticRegression   CodeDataset / HeadTailCodeDataset
                    │                   │
                    ▼                   ▼
              predict_proba      CodeClassifier forward pass
                    │                   │
                    └───────┬───────────┘
                            ▼
                    compute_metrics()
                    (macro-F1, accuracy, precision, recall)
                            │
                            ▼
                    metrics.json + oof.csv + submission.csv
```

## Module boundaries

### `src/data/`

**Responsible for:** loading parquet files, cleaning code text, building PyTorch datasets, creating validation splits.

**Not responsible for:** feature extraction (that's `src/features/`), model architecture.

| File | Input | Output |
|------|-------|--------|
| `preprocess.py` | parquet file paths | DataFrames with `code_length`, `num_lines` added |
| `dataset.py` | list of texts + labels + tokenizer | PyTorch Dataset yielding `{input_ids, attention_mask, labels}` |
| `build_splits.py` | DataFrame | JSON split files (train/val index lists) |

### `src/models/`

**Responsible for:** model architectures and sklearn pipelines.

**Not responsible for:** training loops, data loading, metrics.

| File | What it provides |
|------|-----------------|
| `tfidf_lr.py` | `build_tfidf_lr_pipeline()` → sklearn Pipeline, `predict_proba()` |
| `codebert_cls.py` | `CodeClassifier(backbone)` nn.Module, `load_tokenizer(backbone)` |
| `unixcoder_cls.py` | Re-exports from `codebert_cls.py` (same class, different backbone string in config) |
| `long_context.py` | `sliding_window_inference()` — alternative to HeadTailCodeDataset |
| `hybrid_meta.py` | `build_meta_model()` → CatBoostClassifier, `prepare_meta_features()`, `train_meta_model()` |

### `src/features/`

**Responsible for:** extracting handcrafted features from raw code strings.

**Not responsible for:** feature storage, fusion with neural outputs.

Each module exposes a single function: `extract_*(code: str) -> dict[str, float]`.

| File | Feature count | Examples |
|------|--------------|---------|
| `stylometric.py` | 16 | line length stats, whitespace ratio, indent depth, char composition |
| `structural.py` | 17 | function/class/import counts, bracket/semicolon patterns, comment density |
| `ast_features.py` | 7 | nesting depth, identifier length stats, unique identifier ratio |

**Total: ~40 handcrafted features** when all three are combined.

### `src/runners/`

**Responsible for:** end-to-end training logic per model type — loading data, training, validation, saving artifacts.

| File | Handles runs | Config requirement |
|------|-------------|-------------------|
| `run_tfidf.py` | Run 1 | `model.type: tfidf_lr` |
| `run_transformer.py` | Runs 2/3/4 | `model.architecture: cls_classifier` |
| *(missing)* | Runs 5/6 | `stage1_neural` + `meta_model` — **no runner exists** |

### `src/utils/`

Shared utilities. All modules may import from here.

| File | Key functions |
|------|--------------|
| `io.py` | `load_config()`, `save_json()`, `load_json()`, `ensure_dir()` |
| `seed.py` | `seed_everything(seed)` |
| `logging.py` | `setup_logger(name, log_file)` |
| `metrics.py` | `compute_metrics()`, `compute_per_group_metrics()`, `find_best_threshold()` |

## Config dispatch map

| Config key | Value | Dispatched to | Runs |
|-----------|-------|--------------|------|
| `model.type` | `tfidf_lr` | `run_tfidf.py` | 1 |
| `model.architecture` | `cls_classifier` | `run_transformer.py` | 2, 3, 4 |
| `stage1_neural.*` | *(any)* | **No runner — will crash** | 5, 6 |
| `ensemble.*` | *(any)* | `ensemble.py` (separate entry, stub) | 7 |

## Run status matrix

| Run | Config | Runner | Entry point | Status |
|-----|--------|--------|-------------|--------|
| 1 | ✅ `run1_tfidf_char_baseline_v1.yaml` | ✅ `run_tfidf.py` | `train.py` | **Runnable** |
| 2 | ✅ `run2_codebert_cls_v1.yaml` | ✅ `run_transformer.py` | `train.py` | **Runnable** |
| 3 | ✅ `run3_unixcoder_cls_v1.yaml` | ✅ `run_transformer.py` | `train.py` | **Runnable** |
| 4 | ✅ `run4_unixcoder_longctx_v1.yaml` | ✅ `run_transformer.py` | `train.py` | **Runnable** |
| 5 | ✅ `run5_hybrid_unixcoder_catboost_v1.yaml` | ❌ Missing | — | **Broken** |
| 6 | ✅ `run6_hybrid_robusttrain_v1.yaml` | ❌ Missing | — | **Broken** |
| 7 | ✅ `run7_ensemble_calibrated_v1.yaml` | ❌ Stub | `ensemble.py` | **Stub** |

## Current limitations

1. **Config inheritance is shallow.** `dict.update()` replaces nested dicts entirely. A run config that provides any key under `paths:` must provide all of them.
2. **No distributed training.** Single-GPU only. DataLoader `num_workers=0` is hardcoded in runners.
3. **No model checkpoint loading for inference.** `infer.py` is a stub. After training, you must manually load `best_model.pt`.
4. **Feature extraction is not wired into any pipeline.** The `features/` modules are callable but no runner uses them yet.
5. **No early stopping is implemented** despite `early_stopping_patience: 1` in global config. The transformer runner saves best model but always trains for all epochs.

## Change impact notes

| Change | Impact |
|--------|--------|
| Add a new runner for Run 5/6 | Must register in `train.py` dispatch, or create a new entry point |
| Change `metrics.json` schema | Affects `ensemble.py`, `experiment_tracker.md`, any downstream analysis |
| Modify `CodeClassifier.forward()` return keys | Breaks `run_transformer.py` training loop (expects `{"loss", "logits"}`) |
| Change parquet column names | Breaks `preprocess.py`, `run_tfidf.py`, `run_transformer.py`, `build_splits.py` |
| Add new feature extraction module | Must follow `extract_*(code: str) -> dict` pattern; update `hybrid_meta.py` fusion inputs |
