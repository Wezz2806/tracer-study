"""
Skrip Eksekusi Otomatis Pelatihan Dual-Model IndoBERT Tracer Study.
Melatih Model 1 (Topik Kritik/Saran 5 Kelas) dan Model 2 (Sentimen Testimoni 2 Kelas).
"""

import os
import sys
import json
import pandas as pd
import torch

# Pastikan root workspace terdaftar di sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.train import train_indobert_model, set_seed, get_device
from src.visualize import plot_training_history


def main():
    set_seed(42)
    device = get_device()

    print("\n" + "=" * 80)
    print("🚀 PIPELINE PELATIHAN DUAL-MODEL INDOBERT TRACER STUDY (CRISP-DM FASE 3)")
    print("=" * 80)

    # Pastikan folder output tersedia
    os.makedirs("models/topic_model", exist_ok=True)
    os.makedirs("models/sentiment_model", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # =========================================================================
    # 1. PELATIHAN MODEL 1: TOPIK KRITIK & SARAN (5 KELAS)
    # =========================================================================
    print("\n📦 [1/2] Memuat Dataset Topik...")
    train_topic_path = "data/augmented/train_topic_augmented.csv"
    val_topic_path = "data/processed/val_topic.csv"

    if not os.path.exists(train_topic_path) or not os.path.exists(val_topic_path):
        raise FileNotFoundError("File data latih/validasi topik tidak ditemukan. Pastikan Fase 2 sudah selesai!")

    df_train_topic = pd.read_csv(train_topic_path)
    df_val_topic = pd.read_csv(val_topic_path)

    print(f"• Data Latih Topik   : {len(df_train_topic)} sampel (Augmented)")
    print(f"• Data Validasi Topik: {len(df_val_topic)} sampel (Murni)")

    # Pelatihan Model Topik
    history_topic = train_indobert_model(
        train_df=df_train_topic,
        val_df=df_val_topic,
        task_type="topic",
        model_save_dir="models/topic_model",
        epochs=8,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_length=128,
        patience=3,
        seed=42,
        device=device,
    )

    # Simpan Grafik History Topik
    plot_training_history(
        history=history_topic,
        save_path="reports/figures/training_history_topic.png",
    )

    # =========================================================================
    # 2. PELATIHAN MODEL 2: SENTIMEN TESTIMONI (2 KELAS)
    # =========================================================================
    print("\n📦 [2/2] Memuat Dataset Sentimen...")
    train_sentiment_path = "data/augmented/train_sentiment_augmented.csv"
    val_sentiment_path = "data/processed/val_sentiment.csv"

    if not os.path.exists(train_sentiment_path) or not os.path.exists(val_sentiment_path):
        raise FileNotFoundError("File data latih/validasi sentimen tidak ditemukan.")

    df_train_sentiment = pd.read_csv(train_sentiment_path)
    df_val_sentiment = pd.read_csv(val_sentiment_path)

    print(f"• Data Latih Sentimen   : {len(df_train_sentiment)} sampel (Augmented)")
    print(f"• Data Validasi Sentimen: {len(df_val_sentiment)} sampel (Murni)")

    # Pelatihan Model Sentimen
    history_sentiment = train_indobert_model(
        train_df=df_train_sentiment,
        val_df=df_val_sentiment,
        task_type="sentiment",
        model_save_dir="models/sentiment_model",
        epochs=8,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_length=128,
        patience=3,
        seed=42,
        device=device,
    )

    # Simpan Grafik History Sentimen
    plot_training_history(
        history=history_sentiment,
        save_path="reports/figures/training_history_sentiment.png",
    )

    # =========================================================================
    # 3. REKAPITULASI & PENYIMPANAN LOG
    # =========================================================================
    combined_histories = {
        "topic_model": history_topic,
        "sentiment_model": history_sentiment,
    }

    log_path = "reports/training_histories.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(combined_histories, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🎉 SEMUA MODEL TELAH BERHASIL DILATIH DAN DISIMPAN!")
    print(f"• Checkpoint Model Topik    : models/topic_model/")
    print(f"• Checkpoint Model Sentimen : models/sentiment_model/")
    print(f"• Riwayat Pelatihan (JSON)  : {log_path}")
    print(f"• Grafik Kurva Pelatihan    : reports/figures/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
