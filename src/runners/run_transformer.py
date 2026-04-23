"""Transformer runner (Runs 2/3/4/6)."""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models.codebert_cls import CodeClassifier, load_tokenizer
from src.data.preprocess import load_raw_data, preprocess_dataframe
from src.data.dataset import CodeDataset, HeadTailCodeDataset
from src.utils.metrics import compute_metrics, get_classification_report
from src.utils.io import save_json


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict(model, loader, device, scaler=None):
    """Run inference and return labels (if present), predictions, and logits."""
    model.eval()
    all_labels, all_logits = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels", None)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(**batch)
            else:
                out = model(**batch)

            all_logits.append(out["logits"].cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)           # (N, num_labels)
    preds = logits.argmax(dim=-1).numpy()            # (N,)
    probs = torch.softmax(logits, dim=-1).numpy()    # (N, num_labels)
    labels_np = torch.cat(all_labels).numpy() if all_labels else None

    return labels_np, preds, probs, logits.numpy()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_transformer(cfg: dict, logger):
    paths = cfg["paths"]
    output_dir = cfg["output"]["dir"]
    output_cfg = cfg["output"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_df, val_df, test_df = load_raw_data(paths["train"], paths["test"], paths.get("validation"))
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df, text_col=cfg["data"]["text_column"])

    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_val = val_df[text_col].tolist()
    y_val = val_df[label_col].tolist()
    X_test = test_df[text_col].tolist()

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 2. Tokenizer + Dataset
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(model_cfg["backbone"])
    long_ctx = model_cfg.get("long_context")

    if long_ctx and long_ctx.get("strategy") == "head_tail":
        DatasetCls = HeadTailCodeDataset
        ds_kwargs = {"head_tokens": long_ctx["head_tokens"], "tail_tokens": long_ctx["tail_tokens"]}
    else:
        DatasetCls = CodeDataset
        ds_kwargs = {"max_length": model_cfg["max_length"]}

    train_ds = DatasetCls(X_train, y_train, tokenizer, **ds_kwargs)
    val_ds = DatasetCls(X_val, y_val, tokenizer, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"] * 2, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Model + Optimizer + Scheduler
    # ------------------------------------------------------------------
    model = CodeClassifier(model_cfg["backbone"], num_labels=model_cfg["num_labels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg.get("grad_accum_steps", 1)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if train_cfg.get("mixed_precision") and device == "cuda" else None

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    best_metrics = {}
    grad_accum = train_cfg.get("grad_accum_steps", 1)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            should_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(**batch)
                loss = out["loss"] / grad_accum
                scaler.scale(loss).backward()

                if should_step:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    optimizer.zero_grad()

                    if new_scale >= old_scale:
                        scheduler.step()
            else:
                out = model(**batch)
                loss = out["loss"] / grad_accum
                loss.backward()

                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += out["loss"].item()
            pbar.set_postfix(loss=f"{out['loss'].item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} - Train Loss: {avg_loss:.4f}")

        # --- Validation ---
        val_labels, val_preds, val_probs, val_logits = _predict(model, val_loader, device, scaler)
        metrics = compute_metrics(val_labels, val_preds)
        logger.info(f"Val Epoch {epoch+1}: {metrics}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            logger.info(f"  ↳ Saved best model (F1={best_f1:.4f})")

    logger.info(f"Training complete. Best macro_f1={best_f1:.4f}")

    # ------------------------------------------------------------------
    # 5. Load best checkpoint for post-training inference
    # ------------------------------------------------------------------
    logger.info("Loading best checkpoint for inference...")
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt", map_location=device, weights_only=True))
    model.to(device)

    # ------------------------------------------------------------------
    # 6. OOF predictions on validation set
    # ------------------------------------------------------------------
    val_labels, val_preds, val_probs, val_logits = _predict(model, val_loader, device, scaler)
    best_metrics = compute_metrics(val_labels, val_preds)
    logger.info(f"Best checkpoint validation: {best_metrics}")
    logger.info(f"\n{get_classification_report(val_labels, val_preds)}")

    if output_cfg.get("save_oof", False):
        oof_df = pd.DataFrame({
            "label": val_labels,
            "pred": val_preds,
            "prob_human": val_probs[:, 0],
            "prob_ai": val_probs[:, 1],
        })
        if output_cfg.get("save_logits", False):
            oof_df["logit_0"] = val_logits[:, 0]
            oof_df["logit_1"] = val_logits[:, 1]
        oof_path = f"{output_dir}/oof.csv"
        oof_df.to_csv(oof_path, index=False)
        logger.info(f"Saved OOF predictions → {oof_path} ({len(oof_df):,} rows)")

    # ------------------------------------------------------------------
    # 7. Test predictions + submission
    # ------------------------------------------------------------------
    if output_cfg.get("save_submission", False):
        logger.info("Running inference on test set...")
        test_ds = DatasetCls(X_test, labels=None, tokenizer=tokenizer, **ds_kwargs)
        test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"] * 2, shuffle=False, num_workers=0)

        _, test_preds, test_probs, test_logits = _predict(model, test_loader, device, scaler)

        # Build submission matching sample_submission.csv format (ID, label)
        if "ID" in test_df.columns:
            sub_df = pd.DataFrame({"ID": test_df["ID"].values, "label": test_preds})
        else:
            sub_df = pd.DataFrame({"ID": range(len(test_preds)), "label": test_preds})

        sub_path = f"{output_dir}/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        logger.info(f"Saved submission → {sub_path} ({len(sub_df):,} rows)")

        # Save test logits for ensemble if requested
        if output_cfg.get("save_logits", False):
            logits_df = pd.DataFrame({
                "prob_human": test_probs[:, 0],
                "prob_ai": test_probs[:, 1],
                "logit_0": test_logits[:, 0],
                "logit_1": test_logits[:, 1],
            })
            if "ID" in test_df.columns:
                logits_df.insert(0, "ID", test_df["ID"].values)
            logits_path = f"{output_dir}/test_logits.csv"
            logits_df.to_csv(logits_path, index=False)
            logger.info(f"Saved test logits → {logits_path}")

    # ------------------------------------------------------------------
    # 8. Save final metrics
    # ------------------------------------------------------------------
    save_json({
        "run_name": cfg["run_name"],
        "best_macro_f1": best_f1,
        "validation": {"overall": best_metrics},
    }, f"{output_dir}/metrics.json")

    logger.info("All done ✓")
