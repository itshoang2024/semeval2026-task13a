# Experiment Tracker

| Run | Model | Main change | Random Macro-F1 | Proxy-OOD Lang Macro-F1 | Proxy-OOD Style Macro-F1 | Worst-group F1 | Kaggle Public | Decision | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Run 1 | TF-IDF + LR | char n-gram baseline |  |  |  |  |  | Keep / Drop |  |
| Run 2 | CodeBERT | neural baseline |  |  |  |  |  | Keep / Drop |  |
| Run 3 | UniXcoder | change backbone |  |  |  |  |  | Keep / Drop |  |
| Run 4 | UniXcoder | long-context handling |  |  |  |  |  | Keep / Drop |  |
| Run 5 | UniXcoder + CatBoost | hybrid fusion |  |  |  |  |  | Keep / Drop |  |
| Run 6 | Hybrid | focal + balanced sampler |  |  |  |  |  | Keep / Drop |  |
| Run 7 | Ensemble | calibration + blend |  |  |  |  |  | Final |  |
