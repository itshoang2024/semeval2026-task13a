"""Transformer runner (Runs 2/3/4/6)."""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.models.codebert_cls import CodeClassifier, load_tokenizer
from src.data.preprocess import load_raw_data, preprocess_dataframe
from src.data.dataset import CodeDataset, HeadTailCodeDataset
from src.utils.metrics import compute_metrics
from src.utils.io import save_json


def run_transformer(cfg: dict, logger):
    paths = cfg["paths"]
    output_dir = cfg["output"]["dir"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data (parquet)
    train_df, val_df, test_df = load_raw_data(paths["train"], paths["test"], paths.get("validation"))
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_val = val_df[text_col].tolist()
    y_val = val_df[label_col].tolist()

    # Tokenizer + Dataset
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

    # Model
    model = CodeClassifier(model_cfg["backbone"], num_labels=model_cfg["num_labels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg.get("grad_accum_steps", 1)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if train_cfg.get("mixed_precision") and device == "cuda" else None

    # Training loop
    best_f1 = 0
    grad_accum = train_cfg.get("grad_accum_steps", 1)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            should_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(**batch)
                loss = out["loss"] / grad_accum
                scaler.scale(loss).backward()

                optimizer_stepped = False
                if should_step:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)   # may be skipped internally if grads overflow
                    scaler.update()
                    new_scale = scaler.get_scale()

                    optimizer.zero_grad()

                    # Only step the scheduler if optimizer.step() actually happened
                    if new_scale >= old_scale:
                        scheduler.step()
                        optimizer_stepped = True

                # Optional debug
                # if should_step:
                #     logger.info(f"AMP step={step} old_scale={old_scale} new_scale={new_scale} stepped={optimizer_stepped}")

            else:
                out = model(**batch)
                loss = out["loss"] / grad_accum
                loss.backward()

                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += out["loss"].item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                out = model(**batch)
                preds = out["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        logger.info(f"Val: {metrics}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            logger.info(f"Saved best model (F1={best_f1:.4f})")

    save_json({"run_name": cfg["run_name"], "validation": {"random": metrics}}, f"{output_dir}/metrics.json")
    logger.info("Training complete")
