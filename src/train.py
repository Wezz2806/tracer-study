"""
Modul Pelatihan (Training Engine) untuk Fine-Tuning Dual-Model IndoBERT Tracer Study.
Mendukung optimizer AdamW, Linear Warmup Scheduler, Early Stopping berbasis Macro F1,
dan otomatisasi penyimpanan checkpoint model terbaik.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Optional, Tuple

from src.dataset import create_data_loader
from src.model import (
    get_tokenizer,
    get_model,
    save_model_and_tokenizer,
    TOPIC_ID2LABEL,
    TOPIC_LABEL2ID,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABEL2ID,
    PRETRAINED_MODEL_NAME,
)


def set_seed(seed: int = 42) -> None:
    """
    Menyetel seed random untuk menjamin reproduktifitas hasil eksperimen.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Mendeteksi hardware akselerasi (CUDA GPU jika tersedia, fallback ke CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Menggunakan Akselerasi GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Menggunakan Komputasi CPU")
    return device


def evaluate_epoch(
    model: PreTrainedModel,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluasi model pada satu epoch validasi.

    Returns:
        Tuple (avg_loss, accuracy, macro_precision, macro_recall, macro_f1)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(data_loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return avg_loss, float(acc), float(precision), float(recall), float(f1)


def train_indobert_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task_type: str = "topic",  # "topic" atau "sentiment"
    model_save_dir: str = "models/topic_model",
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
    patience: int = 3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Menjalankan proses fine-tuning IndoBERT secara end-to-end.

    Args:
        train_df: DataFrame data latih (augmented).
        val_df: DataFrame data validasi (murni).
        task_type: Jenis klasifikasi ('topic' atau 'sentiment').
        model_save_dir: Direktori tujuan penyimpanan bobot model terbaik.
        epochs: Jumlah maksimum epoch pelatihan.
        batch_size: Ukuran batch mini.
        learning_rate: Learning rate optimizer AdamW.
        weight_decay: Bobot regularisasi L2.
        warmup_ratio: Proporsi langkah awal linear warmup scheduler.
        max_length: Batas panjang token IndoBERT.
        patience: Batas toleransi early stopping (jumlah epoch tanpa kenaikan Val F1).
        seed: Random seed.
        device: Device target (CPU / CUDA).

    Returns:
        Kamus rekam jejak riwayat pelatihan (history).
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    # 1. Konfigurasi Label berdasarkan Task
    if task_type.lower() == "topic":
        id2label = TOPIC_ID2LABEL
        label2id = TOPIC_LABEL2ID
        task_name = "IndoBERT Topic Classifier (5 Kelas)"
    elif task_type.lower() == "sentiment":
        id2label = SENTIMENT_ID2LABEL
        label2id = SENTIMENT_LABEL2ID
        task_name = "IndoBERT Sentiment Classifier (2 Kelas)"
    else:
        raise ValueError(f"task_type '{task_type}' tidak valid. Pilih 'topic' atau 'sentiment'.")

    num_labels = len(id2label)
    print("\n" + "=" * 70)
    print(f"🔥 MEMULAI FINE-TUNING: {task_name.upper()}")
    print("=" * 70)
    print(f"• Dataset Latih: {len(train_df)} sampel")
    print(f"• Dataset Validasi: {len(val_df)} sampel")
    print(f"• Batch Size: {batch_size} | Max Length: {max_length} | Epochs: {epochs}")
    print(f"• Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"• Early Stopping Patience: {patience} epoch")
    print(f"• Target Simpan: {model_save_dir}")
    print("-" * 70)

    # 2. Inisialisasi Tokenizer & Model
    tokenizer = get_tokenizer(PRETRAINED_MODEL_NAME)
    model = get_model(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        pretrained_name=PRETRAINED_MODEL_NAME,
        classifier_dropout=0.2,
    )
    model.to(device)

    # 3. Buat DataLoader
    train_loader = create_data_loader(
        df=train_df,
        tokenizer=tokenizer,
        text_col="teks_bersih",
        label_col="label",
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
    )

    val_loader = create_data_loader(
        df=val_df,
        tokenizer=tokenizer,
        text_col="teks_bersih",
        label_col="label",
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )

    # 4. Setup Optimizer & Linear Warmup Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 5. Training Loop
    history: Dict[str, Any] = {
        "task_type": task_type,
        "task_name": task_name,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "best_epoch": 0,
        "best_val_macro_f1": 0.0,
        "training_time_seconds": 0.0,
    }

    best_val_f1 = -1.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient Clipping untuk stabilitas gradien Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Evaluasi Validasi
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        epoch_duration = time.time() - epoch_start

        # Simpan History
        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_macro_precision"].append(round(val_prec, 4))
        history["val_macro_recall"].append(round(val_rec, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] -> "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}%"
        )

        # Cek Peningkatan Model Berdasarkan Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            history["best_epoch"] = epoch
            history["best_val_macro_f1"] = round(val_f1, 4)
            epochs_no_improve = 0

            # Simpan Model Terbaik
            metadata = {
                "task_type": task_type,
                "task_name": task_name,
                "best_epoch": epoch,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "weight_decay": weight_decay,
                },
            }
            save_model_and_tokenizer(model, tokenizer, model_save_dir, extra_metadata=metadata)
            print(f"   ⭐ Peningkatan Val Macro F1: {val_f1*100:.2f}%! Model checkpoint disimpan.")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Tidak ada peningkatan ({epochs_no_improve}/{patience} epoch).")

        # Early Stopping Trigger
        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early Stopping diaktifkan pada Epoch {epoch} karena tidak ada peningkatan selama {patience} epoch berturut-turut.")
            break

    total_training_time = time.time() - start_time
    history["training_time_seconds"] = round(total_training_time, 2)

    print("\n" + "=" * 70)
    print(f"🏆 PELATIHAN SELESAI: {task_name}")
    print(f"• Total Waktu: {total_training_time:.2f} detik")
    print(f"• Best Epoch: {history['best_epoch']} | Best Val Macro F1: {history['best_val_macro_f1']*100:.2f}%")
    print("=" * 70 + "\n")

    return history
