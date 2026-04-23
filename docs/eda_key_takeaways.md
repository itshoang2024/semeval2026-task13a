# EDA Key Takeaways — SemEval2026 Task 13A

> After running all cells in the [EDA notebook](../notebooks/eda.ipynb), the following critical observations emerge:

---

## 1. Dataset Overview

| Split | Rows | Columns |
|-------|------|---------|
| **Train** | 500,000 | `code`, `generator`, `label`, `language` |
| **Validation** | 100,000 | `code`, `generator`, `label`, `language` |
| **Test** | 500,000 | `ID`, `code` |

> [!IMPORTANT]
> The test set is **completely blind** — it contains **only `ID` and `code`** columns. No language, no generator information. The model must generalize without knowing the programming language at inference time.

> [!NOTE]
> An official `validation.parquet` (100K rows) exists alongside train and test. It shares the **same schema** as train (`code`, `generator`, `label`, `language`) and can be used directly for model evaluation without needing a custom split from train.

---

## 2. Label Balance — Near Balanced (Good News)

| Split | AI-Generated (label=1) | Human (label=0) | AI Ratio |
|-------|------------------------|-----------------|----------|
| **Train** | 261,525 (52.3%) | 238,475 (47.7%) | 0.523 |
| **Validation** | 52,305 (52.3%) | 47,695 (47.7%) | 0.523 |

- The label ratio is **identical** between train and validation (~52.3/47.7 split).
- The ratio is also **consistent across all three languages** in both splits.
- This near-balance means **no severe class imbalance** needs to be addressed. Standard training procedures should work without heavy resampling.

---

## 3. Language Imbalance — CRITICAL CHALLENGE

### Train

| Language | Count | Percentage |
|----------|-------|-----------|
| **Python** | 457,306 | **91.5%** |
| **C++** | 23,392 | **4.7%** |
| **Java** | 19,302 | **3.9%** |

### Validation

| Language | Count | Percentage |
|----------|-------|-----------|
| **Python** | 91,461 | **91.5%** |
| **C++** | 4,679 | **4.7%** |
| **Java** | 3,860 | **3.9%** |

> [!CAUTION]
> Python **massively dominates** both train and validation (~91.5%). The language proportions are virtually **identical** across splits — the validation set is a proportionally representative sample. This consistency is good for evaluation reliability but the extreme imbalance remains the **#1 challenge** for OOD generalization to minority languages.

**Implications:**
- Stratified sampling across languages is essential
- Language-aware augmentation or weighting strategies should be considered
- Leave-one-language-out validation is critical for OOD evaluation

---

## 4. Generator Diversity — Perfect Overlap

- **35 unique generators** in both train and validation: 34 AI models + `human`
- `human` has 238,475 samples in train, 47,695 in validation (the largest single "generator")
- AI generators have variable sample counts — a **long-tail distribution**
- **Generator overlap is 100%**: all 35 generators appear in both train and val, with **zero exclusive generators** in either direction

> [!NOTE]
> The perfect generator overlap between train and validation means the validation set does **not** test OOD generalization to unseen generators. For robust OOD evaluation, consider leave-one-generator-out or generator-grouped cross-validation strategies.

> [!WARNING]
> At test time, there may be **unseen generators** not present in train/val. Models must learn **generator-invariant** features rather than memorizing specific generator signatures.

---

## 5. Code Length & Line Count — THE STRONGEST SIGNAL

### Code Length Statistics (characters)

| Label | Count | Mean | Std | Min | Median | Q75 | Max |
|-------|-------|------|-----|-----|--------|-----|-----|
| **Human** (0) | 238,475 | **600** | 1,722 | 0 | **319** | 546 | 475,006 |
| **AI** (1) | 261,525 | **1,053** | 894 | 1 | **726** | 1,494 | 11,964 |

### Line Count Statistics

| Label | Count | Mean | Std | Min | Median | Q75 | Max |
|-------|-------|------|-----|-----|--------|-----|-----|
| **Human** (0) | 238,475 | **33** | 84 | 1 | **17** | 28 | 3,775 |
| **AI** (1) | 261,525 | **40** | 35 | 1 | **28** | 51 | 298 |

> [!IMPORTANT]
> **Code length is likely the single strongest feature for classification.** Key patterns:
> - AI-generated code is systematically **longer** than human code (median 726 vs 319 chars)
> - Human code has **huge variance** (std=1,722, max=475K chars) while AI code is more constrained (std=894, max~12K)
> - AI code line counts are more tightly clustered (std=35) vs human code (std=84)
> - Human code has extreme outliers (max 3,775 lines) while AI max is only 298 lines
> - These patterns are **consistent across all three languages** (Python, C++, Java)

---

## 6. Comment Density — Weak Signal

| Metric | Value |
|--------|-------|
| Mean comment density | **3.7%** |
| Median comment density | **0.0%** |
| Q75 comment density | **2.6%** |

- **Most code has zero comments** (median = 0)
- Comment density is a **noisy feature** — useful as part of a feature ensemble but not reliable alone

---

## 7. Whitespace & Style Signals — Moderate Discriminative Power

| Feature | Mean | Std |
|---------|------|-----|
| Whitespace ratio | 0.212 | 0.100 |
| Blank line ratio | 0.136 | 0.116 |
| Mean line length | 22.87 | 56.12 |
| Max line length | 78.96 | 719.95 |
| Mean indentation | 3.12 | 2.97 |

- These stylometric features show moderate variance between Human and AI classes
- They have discriminative power **when combined** in an ensemble but are individually moderate signals

---

## 8. Semicolon & Brace Counts — Language Indicators

| Feature | Mean | Q25-Q75 |
|---------|------|---------|
| Semicolons | 3.71 | **0 – 0** |
| Braces (`{` + `}`) | 3.22 | **0 – 0** |

- These are primarily **language indicators** rather than label indicators
- Python code has ~0 semicolons and braces, while C++/Java use them heavily
- Useful for distinguishing programming language, which **indirectly helps** classification

---

## 9. Train vs Test Distribution Shift — Minimal

- Train: mean=839 chars, median=489 chars, max=475,006 chars
- Test: similar distribution characteristics (based on histogram overlap)
- Log code length and line count histograms show **good overlap** between train and test
- **No dramatic covariate shift** in code length is observed, which is encouraging for model deployment

---

## 10. Train vs Validation — Nearly Identical Distributions

Key comparisons between train and validation splits:

| Metric | Train | Validation | Difference |
|--------|-------|------------|-----------|
| AI ratio | 0.5231 | 0.5231 | ~0.0000 |
| Python % | 91.5% | 91.5% | ~0.0% |
| C++ % | 4.7% | 4.7% | ~0.0% |
| Java % | 3.9% | 3.9% | ~0.0% |
| Generator count | 35 | 35 | 0 |
| Generator overlap | — | — | **100%** |

> [!NOTE]
> The validation set appears to be a **perfectly stratified sample** from the same distribution as train. Label ratios, language proportions, and generator sets are virtually identical. This means:
> - Validation performance should be a **reliable estimate** of train-distribution performance
> - However, it does **not** measure OOD robustness (unseen languages, generators, or domains)
> - For OOD evaluation, use leave-one-language-out or leave-one-generator-out strategies

---

## 11. Modeling Implications

1. **Code length / line count features are strong baselines** — even simple thresholding on length could achieve reasonable accuracy
2. **The extreme Python dominance requires careful handling** — consider leave-one-language-out validation for honest OOD evaluation
3. **Head-tail truncation strategies** (HeadTailCodeDataset) are justified by the long-tail length distribution of human code
4. **Stylometric feature ensembles** (whitespace, indentation, comments) add incremental value beyond length alone
5. **The 34 AI generators create a long-tail OOD challenge** — models must learn generator-invariant patterns, not generator-specific signatures
6. **Test set blindness** (no language/generator info) means the model must be robust without metadata
7. **The official validation set is reliable** for in-distribution evaluation, but custom OOD splits (language-out, generator-out) are still essential for robustness testing
8. **Perfect generator overlap** between train and val means val alone cannot assess generalization to unseen generators — design OOD experiments accordingly
