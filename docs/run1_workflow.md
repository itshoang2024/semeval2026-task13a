## Run 1 — TF-IDF char n-gram + LogReg: Execution Flow

```
python src/train.py --config configs/run1_tfidf_char_baseline_v1.yaml
```

### Trình tự chạy:

```mermaid
flowchart TD
    A["train.py<br/>Load config + merge global.yaml"] --> B["seed_everything(42)"]
    B --> C["Create output dir<br/>runs/run1_tfidf_char_baseline_v1/"]
    C --> D["Setup logger → train.log"]
    D --> E["Copy config → output dir"]
    E --> F{"model.type == tfidf_lr"}
    F --> G["Dispatch → run_tfidf()"]
    
    G --> H{"data/processed/*.parquet<br/>exists?"}
    H -- Yes --> I["Read processed parquet<br/>(skip preprocessing)"]
    H -- No --> J["Read raw parquet<br/>+ preprocess_dataframe()"]
    
    I --> K["Build TF-IDF + LogReg Pipeline"]
    J --> K
    
    K --> L["pipeline.fit(X_train, y_train)<br/>500K samples"]
    L --> M["predict_proba(X_val)<br/>100K samples"]
    M --> N["compute_metrics()<br/>macro-F1, accuracy, precision, recall"]
    N --> O["Save metrics.json + oof.csv"]
    O --> P["predict_proba(X_test)<br/>500K samples"]
    P --> Q["Save submission.csv"]
```

### Chi tiết từng bước:

| # | Bước | Mô tả |
|---|------|--------|
| 1 | **Config merge** | `run1_*.yaml` inherits `global.yaml` via `dict.update()` — run config overrides global defaults |
| 2 | **Seed** | Fix random state (42) cho reproducibility |
| 3 | **Data loading** | **[MỚI]** Ưu tiên `data/processed/{train,val,test}_clean.parquet`. Nếu không có → fallback load raw + `preprocess_dataframe()` |
| 4 | **TF-IDF vectorizer** | `char_wb` analyzer, ngram (3,5), max 200K features, sublinear TF |
| 5 | **LogReg fit** | C=4.0, balanced class weight, LBFGS solver, max 1000 iter, n_jobs=-1 |
| 6 | **Validation** | Predict probabilities → threshold 0.5 → compute macro-F1 |
| 7 | **Save outputs** | `metrics.json`, `oof.csv` (val predictions), `submission.csv` (test predictions) |

### Output artifacts:
```
runs/run1_tfidf_char_baseline_v1/
├── config.yaml        # copy of config used
├── train.log          # full training log
├── metrics.json       # {"run_name": ..., "validation": {"random": {macro_f1, accuracy, ...}}}
├── oof.csv            # prob, pred, true for 100K val samples
└── submission.csv     # ID, label for 500K test samples
```