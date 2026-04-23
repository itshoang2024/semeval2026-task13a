# Data Processing Pipeline - Key Takeaways

> Extracted from the outputs of [`notebooks/data_processing.ipynb`](../notebooks/data_processing.ipynb).

---

## 1. Data Quality - Very Clean

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| **Raw rows** | 500,000 | 100,000 | 500,000 |
| **After cleaning** | 499,896 | 100,000 | 500,000 |
| **Dropped** | 104 (0.02%) | 0 | 0 |

- Chỉ **104/500,000 rows** (0.02%) bị drop - dữ liệu gốc đã rất sạch.
- Thành phần bị drop: **1 sample rỗng** (label=0) + **103 samples "binary-ish"** (chứa >20% ký tự `\xa0` non-breaking spaces - tất cả đều label=1, đến từ AI generators dùng NBSP thay cho tab/space).
- Val và test: **0 samples bị drop** - sạch hoàn toàn.

---

## 2. Outlier Flags - Hầu hết là extreme length

| Flag | Train | Val | Test |
|------|-------|-----|------|
| `is_empty` | 1 | 0 | 0 |
| `is_binary_ish` | 103 | 27 | 11 |
| `is_minified` | 68 | 6 | 8 |
| `is_extreme_length` | 4,999 | 1,000 | 4,999 |
| **Total flagged** | 5,169 (~1%) | 1,033 (~1%) | 5,018 (~1%) |

### Quan sát đáng chú ý:

- **`is_binary_ish`**: Chủ yếu là AI-generated code sử dụng `\xa0` (non-breaking space) thay indent thường. Mẫu code thực chất hợp lệ về mặt logic nhưng encoding bất thường → quyết định drop khỏi train.
- **`is_minified`**: 68 samples trong train - đáng chú ý là nhiều sample **không phải code thật** mà là **text mô tả bằng tiếng Anh** ("In this solution, the function...", "Tags: `Adobe Premiere Pro`..."). Tất cả đều label=1 → AI đôi khi sinh ra giải thích thay vì code. Không drop ở vòng này, chỉ flag.
- **`is_extreme_length`**: ~1% top percentile mỗi split, tương đương thiết kế (99th percentile). Không cần drop - transformer truncation xử lý.

---

## 3. Feature Extraction - 40 Features, ~13 phút tổng

| Split | Rows | Features | Time |
|-------|------|----------|------|
| Train | 499,896 | 40 | 4m50s (~1,719 it/s) |
| Val | 100,000 | 40 | 1m00s (~1,664 it/s) |
| Test | 500,000 | 40 | 7m08s (~1,166 it/s) |

### 40 features chia 3 nhóm:

| Nhóm | Số features | Ví dụ |
|------|-------------|-------|
| **Stylometric** (16) | Dạng text/format | `mean_line_length`, `whitespace_ratio`, `tab_ratio`, `alpha_ratio`, `uppercase_ratio` |
| **Structural** (17) | Dạng cấu trúc code | `num_functions`, `num_classes`, `semicolon_count`, `brace_count`, `comment_density` |
| **AST-lite** (7) | Dạng depth/identifier | `max_nesting_depth`, `num_identifiers`, `mean_identifier_length`, `unique_identifier_ratio` |

- Test set chậm hơn (~1,166 it/s vs ~1,700 it/s) - có thể do test chứa code dài hơn trung bình (không có label filter).
- Tất cả features đã lưu tại `data/features/handcrafted_*.parquet`, sẵn sàng cho Run 1 (TF-IDF) và Run 5+ (hybrid meta).

---

## 4. Validation Splits - Stratification Chính Xác

### 4 loại splits đã tạo:

| Split Type | Mục đích | Files |
|------------|----------|-------|
| **Holdout** (90/10) | Quick ablation | 449,906 train / 49,990 val |
| **10-fold CV** | Robust estimate | ~449,906 train / ~49,990 val per fold |
| **LOLO** (Leave-one-lang-out) | OOD language test | 3 splits |
| **Length bins** | OOD length test | 3 splits |

### LOLO (quan trọng nhất cho OOD):

| Held-out Language | Train size | Val size | Ghi chú |
|-------------------|-----------|----------|---------|
| Python | 42,677 | 457,219 | ⚠️ Train chỉ ~43K khi bỏ Python |
| Java | 480,605 | 19,291 | Val nhỏ, cẩn thận variance |
| C++ | 476,510 | 23,386 | Val nhỏ, cẩn thận variance |

### Kiểm chứng stratification (holdout split):

| Metric | Train | Val | Δ |
|--------|-------|-----|---|
| AI ratio | 0.5230 | 0.5230 | ~0.0000 |
| C++ % | 4.68% | 4.68% | ~0.00% |
| Java % | 3.86% | 3.86% | ~0.00% |
| Python % | 91.46% | 91.46% | ~0.00% |
| C++ AI ratio | 0.5234 | 0.5233 | 0.0001 |
| Java AI ratio | 0.5218 | 0.5220 | 0.0002 |
| Python AI ratio | 0.5230 | 0.5230 | 0.0000 |

> **Stratification hoàn hảo** - cả label lẫn language đều giữ đúng tỷ lệ giữa train/val trong holdout split. Fix `label × language` đã hoạt động chính xác.

---

## 5. Output Artifacts

| Đường dẫn | Kích thước | Mô tả |
|-----------|-----------|-------|
| `data/processed/train_clean.parquet` | 193.0 MB | 499,896 rows (sau drop pathological) |
| `data/processed/val_clean.parquet` | 38.5 MB | 100,000 rows (lockbox) |
| `data/processed/test_clean.parquet` | 281.3 MB | 500,000 rows (blind) |
| `data/features/handcrafted_train.parquet` | 39.2 MB | 40 features × 499,896 rows |
| `data/features/handcrafted_val.parquet` | 8.7 MB | 40 features × 100,000 rows |
| `data/features/handcrafted_test.parquet` | 44.8 MB | 40 features × 500,000 rows |
| `data/splits/holdout_v1.json` | 7.0 MB | Single 90/10 split |
| `data/splits/random_stratified_v1.json` | 70.5 MB | 10-fold CV indices |
| `data/splits/lolo_language_v1.json` | 21.1 MB | Leave-one-language-out |
| `data/splits/length_style_bins_v1.json` | 21.1 MB | Length bin holdouts |

---

## 6. Implications cho Training

1. **Data gần như không cần thêm cleaning** - chỉ 0.02% bị loại. Pipeline đang production-ready.
2. **`is_minified` samples đáng quan tâm** - 68 samples "minified" trong train thực chất là text mô tả, không phải code. Cân nhắc flag riêng cho "non-code text" samples trong vòng sau.
3. **LOLO-Python split cực kỳ khắc nghiệt** - khi holdout Python, train chỉ còn ~43K samples (C++ + Java). Model phải học cross-language features từ lượng data rất nhỏ.
4. **40 handcrafted features sẵn sàng** - có thể dùng ngay cho Run 1 (TF-IDF + LogReg) và Run 5 (hybrid CatBoost meta-model).
5. **Stratification đã fix thành công** - không còn bug label-only. Mọi split giờ đều giữ đúng tỷ lệ `label × language`.
