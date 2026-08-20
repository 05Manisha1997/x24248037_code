from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train_transformer(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    output_dir: str,
    num_labels: int = 2,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    return {
        "model_name": model_name,
        "train_loss": float(train_result.training_loss),
        "eval": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "trainer": trainer,
    }


def predict_transformer(trainer: Trainer, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    ds = TextDataset(texts, [0] * len(texts), trainer.tokenizer if hasattr(trainer, "tokenizer") else None)
    # Re-tokenize via trainer's model tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trainer.model.config._name_or_path)
    ds = TextDataset(texts, [0] * len(texts), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size)
    trainer.model.eval()
    all_preds, all_probs = [], []
    device = trainer.model.device
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = trainer.model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] == 2 else probs.max(dim=-1).values.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)
