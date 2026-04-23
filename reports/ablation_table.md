# Ablation Table

| ID | Base model | Long context | Handcrafted features | Focal loss | Balanced sampler | Ensemble | Proxy-OOD Macro-F1 | Kaggle Public | Conclusion |
|---|---|---|---|---|---|---|---:|---:|---|
| A1 | CodeBERT | No | No | No | No | No |  |  |  |
| A2 | UniXcoder | No | No | No | No | No |  |  |  |
| A3 | UniXcoder | Yes | No | No | No | No |  |  |  |
| A4 | UniXcoder | Yes | Yes | No | No | No |  |  |  |
| A5 | UniXcoder | Yes | Yes | Yes | Yes | No |  |  |  |
| A6 | UniXcoder | Yes | Yes | Yes | Yes | Yes |  |  |  |
