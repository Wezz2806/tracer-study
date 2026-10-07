"""
Modul Evaluasi Model untuk Sistem NLP Tracer Study IndoBERT.
Menghitung metrik performa standar industri (Accuracy, Macro Precision, Macro Recall, Macro F1,
Confusion Matrix, dan Classification Report) pada Test Set murni tanpa data leakage.
"""

import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.dataset import create_data_loader
from src.train import get_device


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_df: pd.DataFrame,
    id2label: Dict[int, str],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Menjalankan pengujian performa model pada Test Set.

    Args:
        model: Model IndoBERT terlatih.
        tokenizer: Tokenizer IndoBERT.
        test_df: DataFrame dataset uji (murni).
        id2label: Kamus pemetaan ID ke nama label.
        device: Device komputasi (CPU / CUDA).
        batch_size: Ukuran batch.
        max_length: Batas panjang token.

    Returns:
        Kamus berisi metrik evaluasi lengkap dan DataFrame hasil prediksi.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    test_loader = create_data_loader(
        df=test_df,
        tokenizer=tokenizer,
        text_col="teks_bersih",
        label_col="label",
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_confidences: List[float] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # Komputasi Metrik Evaluasi
    acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Confusion Matrix
    unique_labels = sorted(list(id2label.keys()))
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    # Classification Report
    target_names = [id2label[i] for i in unique_labels]
    clf_report_dict = classification_report(
        all_labels,
        all_preds,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    clf_report_text = classification_report(
        all_labels,
        all_preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )

    # Buat DataFrame Detail Prediksi
    preds_df = test_df.copy()
    preds_df["pred_label"] = all_preds
    preds_df["pred_label_name"] = [id2label[p] for p in all_preds]
    preds_df["confidence"] = [round(c, 4) for c in all_confidences]
    preds_df["is_correct"] = preds_df["label"] == preds_df["pred_label"]

    results = {
        "total_samples": len(test_df),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels,
        "label_names": target_names,
        "classification_report_dict": clf_report_dict,
        "classification_report_text": clf_report_text,
        "predictions_df": preds_df,
    }

    return results


def print_evaluation_summary(results: Dict[str, Any], task_title: str) -> None:
    """
    Mencetak ringkasan hasil evaluasi ke terminal secara terstruktur.
    """
    print("\n" + "=" * 75)
    print(f"📊 HASIL EVALUASI TEST SET: {task_title.upper()}")
    print("=" * 75)
    print(f"• Total Data Uji    : {results['total_samples']} sampel (Murni Tanpa Augmentasi)")
    print(f"• Accuracy          : {results['accuracy']*100:.2f}%")
    print(f"• Macro Precision   : {results['macro_precision']*100:.2f}%")
    print(f"• Macro Recall      : {results['macro_recall']*100:.2f}%")
    print(f"• Macro F1-Score    : {results['macro_f1']*100:.2f}%")
    print(f"• Weighted F1-Score : {results['weighted_f1']*100:.2f}%")
    print("-" * 75)
    print("📋 Classification Report Per-Kelas:")
    print(results["classification_report_text"])
    print("=" * 75 + "\n")
