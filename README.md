# SemEval2026 Task 13A — Machine-Generated Code Detection

Binary classification: human-written vs. AI-generated code.

## Strategy

UniXcoder-centered hybrid ensemble with OOD-first training.  

## Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Data (already in data/raw/)
#    train.parquet, validation.parquet, test.parquet, sample_submission.csv

# 3. Run a training experiment (Runs 1-4 are functional)
python src/train.py --config configs/run1_tfidf_char_baseline_v1.yaml

# 4. Inference & ensemble — not yet implemented (stubs)
# python src/infer.py --config ...
# python src/ensemble.py --config ...
```

## Project Structure

```
├── configs/          # YAML configs (global + per-run)
├── data/raw/         # Official parquet data
├── src/              # Training, inference, models, features, utils
├── runs/             # Experiment outputs (metrics, OOF, submissions)
├── reports/          # Experiment tracker, ablation, Kaggle log
├── notebooks/        # EDA, error analysis
└── docs/             # Reference papers, architecture doc
```

See [docs/architecture.md](docs/architecture.md) for module boundaries, data flow, and run status.

## 7-Run Roadmap

| Run | Model | Focus | Status |
|-----|-------|-------|--------|
| 1 | TF-IDF + LR | Baseline, first Kaggle submission | ✅ |
| 2 | CodeBERT | Neural baseline | ✅ |
| 3 | UniXcoder | Backbone selection | ✅ |
| 4 | UniXcoder + long-ctx | Input handling | ✅ |
| 5 | Hybrid (UniXcoder + CatBoost) | Feature fusion | ❌ needs runner |
| 6 | Hybrid + robust training | Focal loss, balanced sampling | ❌ needs runner |
| 7 | Ensemble + calibration | Final submission | ❌ stub |

## Key Decisions

- **Metric**: macro-F1
- **Validation**: random split + leave-one-language-out + length/style bins
- **Seed**: 42
- **No extra training data, no pretrained AI-code detectors**

