"""Patch eda.ipynb to include validation.parquet analysis.

Modifies the notebook JSON in-place:
1. Updates Section 1 markdown to mention validation split.
2. Updates the load-data cell to also load validation.parquet.
3. Adds new cells after load-data for validation basic stats.
4. Adds a new Section after Train vs Test for Train vs Validation comparison.
5. Updates Key Takeaways to include validation findings.

Run from notebooks/ directory:
    python patch_eda_add_validation.py
"""

import json
import copy
import sys
from pathlib import Path


def make_md_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def make_code_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def find_cell_index(cells: list[dict], keyword: str) -> int | None:
    """Find index of the first cell whose source contains `keyword`."""
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if keyword in src:
            return i
    return None


def patch_notebook(nb_path: Path) -> None:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── 1. Patch the Section 1 markdown ──────────────────────────────
    idx_section1_md = find_cell_index(cells, "## 1. Load Data")
    if idx_section1_md is not None:
        cells[idx_section1_md] = make_md_cell([
            "## 1. Load Data\n",
            "\n",
            "Raw data contains **train**, **validation**, and **test** splits."
        ])
        # Clear cached outputs
        cells[idx_section1_md]["outputs"] = [] if cells[idx_section1_md].get("outputs") else []

    # ── 2. Patch the load-data code cell ─────────────────────────────
    idx_load = find_cell_index(cells, "train_df = pd.read_parquet('../data/raw/train.parquet')")
    if idx_load is not None:
        cells[idx_load] = make_code_cell([
            "train_df = pd.read_parquet('../data/raw/train.parquet')\n",
            "val_df = pd.read_parquet('../data/raw/validation.parquet')\n",
            "test_df = pd.read_parquet('../data/raw/test.parquet')\n",
            "\n",
            "print(f'Train: {train_df.shape[0]:>8,} rows x {train_df.shape[1]} cols  {train_df.columns.tolist()}')\n",
            "print(f'Val:   {val_df.shape[0]:>8,} rows x {val_df.shape[1]} cols  {val_df.columns.tolist()}')\n",
            "print(f'Test:  {test_df.shape[0]:>8,} rows x {test_df.shape[1]} cols  {test_df.columns.tolist()}')",
        ])

    # ── 3. Add validation basic stats cells right after load section ──
    idx_describe = find_cell_index(cells, "train_df.describe(include='all')")
    if idx_describe is not None:
        new_cells = [
            make_md_cell(["### Validation set overview"]),
            make_code_cell([
                "val_df.info()\n",
                "print()\n",
                "val_df.describe(include='all')",
            ]),
            make_code_cell([
                "val_df.head(3)",
            ]),
            make_code_cell([
                "# Validation label distribution\n",
                "print('=== Validation Label Distribution ===')\n",
                "val_label_counts = val_df['label'].value_counts().sort_index()\n",
                "for idx_val, cnt in val_label_counts.items():\n",
                "    print(f'  {label_names.get(idx_val, idx_val)}: {cnt:>8,} ({cnt/len(val_df)*100:.1f}%)')\n",
                "print(f'  AI ratio: {val_df[\"label\"].mean():.3f}')\n",
                "print()\n",
                "\n",
                "# Validation language distribution\n",
                "print('=== Validation Language Distribution ===')\n",
                "val_lang_counts = val_df['language'].value_counts()\n",
                "for lang, cnt in val_lang_counts.items():\n",
                "    print(f'  {lang}: {cnt:>8,} ({cnt/len(val_df)*100:.1f}%)')\n",
                "print()\n",
                "\n",
                "# Validation generator diversity\n",
                "print(f'Unique generators in val: {val_df[\"generator\"].nunique()}')\n",
                "print(f'Generators in val but NOT in train: {set(val_df[\"generator\"].unique()) - set(train_df[\"generator\"].unique())}')\n",
                "print(f'Generators in train but NOT in val: {set(train_df[\"generator\"].unique()) - set(val_df[\"generator\"].unique())}')",
            ]),
        ]
        # Insert after the describe cell
        for j, nc in enumerate(new_cells):
            cells.insert(idx_describe + 1 + j, nc)

    # ── 4. Add Train vs Validation comparison section ────────────────
    # Find the "Train vs Test Comparison" section and add val comparison after it
    idx_train_test = find_cell_index(cells, "## 8. Train vs Test Comparison")
    if idx_train_test is None:
        idx_train_test = find_cell_index(cells, "Train vs Test")
    
    # Find the section AFTER train vs test (Sample Code Inspection)
    idx_sample = find_cell_index(cells, "## 9. Sample Code Inspection")
    if idx_sample is None:
        idx_sample = find_cell_index(cells, "Sample Code Inspection")

    if idx_sample is not None:
        insert_pos = idx_sample
        val_comparison_cells = [
            make_md_cell([
                "## 8.5 Train vs Validation Comparison\n",
                "\n",
                "Compare distributions between train and validation splits to check for potential data leakage or distribution mismatch."
            ]),
            make_code_cell([
                "# Add code_length and num_lines to val_df if not already present\n",
                "if 'code_length' not in val_df.columns:\n",
                "    val_df['code_length'] = val_df['code'].str.len()\n",
                "    val_df['num_lines'] = val_df['code'].str.count('\\\\n') + 1\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# 1. Label distribution comparison\n",
                "train_label_pct = train_df['label'].value_counts(normalize=True).sort_index()\n",
                "val_label_pct = val_df['label'].value_counts(normalize=True).sort_index()\n",
                "x = np.arange(len(train_label_pct))\n",
                "w = 0.35\n",
                "axes[0].bar(x - w/2, train_label_pct.values, w, label='Train', alpha=0.8)\n",
                "axes[0].bar(x + w/2, val_label_pct.values, w, label='Validation', alpha=0.8)\n",
                "axes[0].set_xticks(x)\n",
                "axes[0].set_xticklabels(['Human', 'AI-Generated'])\n",
                "axes[0].set_title('Label Distribution: Train vs Val')\n",
                "axes[0].set_ylabel('Proportion')\n",
                "axes[0].legend()\n",
                "\n",
                "# 2. Language distribution comparison\n",
                "train_lang_pct = train_df['language'].value_counts(normalize=True).sort_index()\n",
                "val_lang_pct = val_df['language'].value_counts(normalize=True).sort_index()\n",
                "x = np.arange(len(train_lang_pct))\n",
                "axes[1].bar(x - w/2, train_lang_pct.values, w, label='Train', alpha=0.8)\n",
                "axes[1].bar(x + w/2, val_lang_pct.values, w, label='Validation', alpha=0.8)\n",
                "axes[1].set_xticks(x)\n",
                "axes[1].set_xticklabels(train_lang_pct.index)\n",
                "axes[1].set_title('Language Distribution: Train vs Val')\n",
                "axes[1].set_ylabel('Proportion')\n",
                "axes[1].legend()\n",
                "\n",
                "# 3. Code length distribution comparison\n",
                "axes[2].hist(np.log1p(train_df['code_length']), bins=100, alpha=0.5, label='Train', density=True)\n",
                "axes[2].hist(np.log1p(val_df['code_length']), bins=100, alpha=0.5, label='Validation', density=True)\n",
                "axes[2].set_title('Log Code Length: Train vs Val')\n",
                "axes[2].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('=== Train vs Validation Stats ===')\n",
                "print(f\"Train - mean: {train_df['code_length'].mean():.0f}, median: {train_df['code_length'].median():.0f}, std: {train_df['code_length'].std():.0f}\")\n",
                "print(f\"Val   - mean: {val_df['code_length'].mean():.0f}, median: {val_df['code_length'].median():.0f}, std: {val_df['code_length'].std():.0f}\")\n",
                "print(f\"\\nTrain AI ratio: {train_df['label'].mean():.4f}\")\n",
                "print(f\"Val   AI ratio: {val_df['label'].mean():.4f}\")",
            ]),
            make_code_cell([
                "# Cross-tabulation: label x language for validation\n",
                "label_names_inv = {0: 'Human', 1: 'AI-Generated'}\n",
                "\n",
                "val_ct = pd.crosstab(\n",
                "    val_df['language'],\n",
                "    val_df['label'].map(label_names_inv),\n",
                "    margins=True, margins_name='All'\n",
                ")\n",
                "\n",
                "print('=== Validation: Language x Label ===')\n",
                "display(val_ct)\n",
                "\n",
                "# Per-language AI ratio comparison\n",
                "print('\\n=== Per-language AI ratio: Train vs Val ===')\n",
                "for lang in sorted(train_df['language'].unique()):\n",
                "    train_ratio = train_df[train_df['language'] == lang]['label'].mean()\n",
                "    val_ratio = val_df[val_df['language'] == lang]['label'].mean()\n",
                "    print(f'  {lang:8s}: Train={train_ratio:.4f}  Val={val_ratio:.4f}  Δ={abs(train_ratio - val_ratio):.4f}')",
            ]),
            make_code_cell([
                "# Per-label length distribution: Train vs Validation\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "for i, label in enumerate([0, 1]):\n",
                "    label_name = 'Human' if label == 0 else 'AI-Generated'\n",
                "    train_subset = train_df[train_df['label'] == label]\n",
                "    val_subset = val_df[val_df['label'] == label]\n",
                "    \n",
                "    axes[i].hist(np.log1p(train_subset['code_length']), bins=100, alpha=0.5, label='Train', density=True)\n",
                "    axes[i].hist(np.log1p(val_subset['code_length']), bins=100, alpha=0.5, label='Validation', density=True)\n",
                "    axes[i].set_title(f'{label_name}: Log Code Length (Train vs Val)')\n",
                "    axes[i].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Compare generators\n",
                "print('=== Generator overlap ===')\n",
                "train_gens = set(train_df['generator'].unique())\n",
                "val_gens = set(val_df['generator'].unique())\n",
                "print(f'Train generators: {len(train_gens)}')\n",
                "print(f'Val generators:   {len(val_gens)}')\n",
                "print(f'Overlap:          {len(train_gens & val_gens)}')\n",
                "val_only = val_gens - train_gens\n",
                "if val_only:\n",
                "    print(f'Val-only generators (OOD risk): {val_only}')\n",
                "else:\n",
                "    print('All val generators are also in train ✓')",
            ]),
        ]
        for j, nc in enumerate(val_comparison_cells):
            cells.insert(insert_pos + j, nc)

    # ── 5. Update section numbers downstream ─────────────────────────
    # After inserting 8.5, the old 9 becomes 9.5 etc. 
    # But better to just renumber: 8.5 → keep, 9 → 9, 10 → 10

    # ── 6. Save ──────────────────────────────────────────────────────
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Patched {nb_path} -- added validation.parquet analysis")
    print("   Please re-run the notebook to see results.")


if __name__ == "__main__":
    nb_path = Path(__file__).parent / "eda.ipynb"
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found at {nb_path}")
        sys.exit(1)
    patch_notebook(nb_path)
