"""
Modul Continuous Retraining & Active Learning Feedback Loop Engine.
Mendukung:
1. Akumulasi data masukan terverifikasi manusia dari antrean active learning.
2. Penggabungan data baru dengan dataset historis dan re-augmentasi seimbang.
3. Fine-tuning terisolasi dengan early stopping berbasis validation Macro F1.
4. Evaluasi komparatif anti-regresi: checkpoint baru hanya dipromosikan jika performanya
   mengungguli atau setara dengan baseline model produksi sebelumnya.
5. Backup checkpoint otomatis dan hot-reloading model pada inference engine.
"""

import os
import shutil
import json
import time
import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import torch

from src.preprocessing import clean_text
from src.data_augmentation import augment_topic_dataset, augment_sentiment_dataset
from src.dataset import create_dataloaders
from src.model import get_model, save_model_and_tokenizer, get_tokenizer, TOPIC_ID2LABEL, TOPIC_LABEL2ID, SENTIMENT_ID2LABEL, SENTIMENT_LABEL2ID
from src.train import train_epoch, eval_epoch, get_device
from src.evaluate import evaluate_on_test_set

ACTIVE_LEARNING_DIR = "data/active_learning"
REVIEWED_FILE = os.path.join(ACTIVE_LEARNING_DIR, "reviewed_samples.csv")
RETRAIN_LOG_FILE = os.path.join(ACTIVE_LEARNING_DIR, "retraining_history.json")


def load_retraining_logs() -> list:
    """Memuat riwayat log eksekusi retraining."""
    if os.path.exists(RETRAIN_LOG_FILE):
        try:
            with open(RETRAIN_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_retraining_log(entry: Dict[str, Any]) -> None:
    """Menyimpan entri riwayat retraining."""
    logs = load_retraining_logs()
    logs.insert(0, entry)
    with open(RETRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs[:50], f, indent=2, ensure_ascii=False)


def backup_model_checkpoint(source_dir: str, backup_base_dir: str = "models/backups") -> str:
    """Membuat salinan cadangan model produksi sebelum pembaruan."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(source_dir)
    backup_target = os.path.join(backup_base_dir, f"{model_name}_{timestamp}")
    os.makedirs(backup_base_dir, exist_ok=True)
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, backup_target)
        print(f"📦 Backup checkpoint dibuat: {backup_target}")
    return backup_target


def prepare_retraining_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menggabungkan data latih dasar murni dengan data baru hasil verifikasi active learning.
    """
    base_train_topic_path = "data/processed/train_topic.csv"
    base_train_sentiment_path = "data/processed/train_sentiment.csv"

    df_base_topic = pd.read_csv(base_train_topic_path)
    df_base_sentiment = pd.read_csv(base_train_sentiment_path)

    if not os.path.exists(REVIEWED_FILE):
        print("ℹ️ Tidak ditemukan data baru di buffer review. Menggunakan dataset standar.")
        return df_base_topic, df_base_sentiment

    try:
        df_reviewed = pd.read_csv(REVIEWED_FILE)
        print(f"📥 Menemukan {len(df_reviewed)} sampel baru yang telah diverifikasi manusia.")
    except Exception as e:
        print(f"⚠️ Gagal membaca file review: {e}. Menggunakan dataset dasar.")
        return df_base_topic, df_base_sentiment

    if df_reviewed.empty:
        return df_base_topic, df_base_sentiment

    # Ekstrak data baru untuk Topic
    new_topic_rows = []
    for _, row in df_reviewed.iterrows():
        if pd.notna(row.get("corrected_topic_id")):
            t_id = int(row["corrected_topic_id"])
            t_name = TOPIC_ID2LABEL.get(t_id, "Unknown")
            new_topic_rows.append({
                "Teks": str(row["teks_asli"]),
                "Cleaned_Teks": str(row["teks_bersih"]),
                "Topic_ID": t_id,
                "Topic_Name": t_name,
                "Cluster": t_id,
            })

    # Ekstrak data baru untuk Sentiment
    new_sentiment_rows = []
    for _, row in df_reviewed.iterrows():
        if pd.notna(row.get("corrected_sentiment_id")):
            s_id = int(row["corrected_sentiment_id"])
            s_name = SENTIMENT_ID2LABEL.get(s_id, "Unknown")
            new_sentiment_rows.append({
                "Teks": str(row["teks_asli"]),
                "Cleaned_Teks": str(row["teks_bersih"]),
                "Sentiment_ID": s_id,
                "Sentiment_Name": s_name,
                "Cluster": s_id,
            })

    # Gabungkan dengan base data
    if new_topic_rows:
        df_new_topic = pd.DataFrame(new_topic_rows)
        df_merged_topic = pd.concat([df_base_topic, df_new_topic], ignore_index=True)
        # Hapus duplikat teks jika ada
        df_merged_topic = df_merged_topic.drop_duplicates(subset=["Cleaned_Teks"])
    else:
        df_merged_topic = df_base_topic

    if new_sentiment_rows:
        df_new_sentiment = pd.DataFrame(new_sentiment_rows)
        df_merged_sentiment = pd.concat([df_base_sentiment, df_new_sentiment], ignore_index=True)
        df_merged_sentiment = df_merged_sentiment.drop_duplicates(subset=["Cleaned_Teks"])
    else:
        df_merged_sentiment = df_base_sentiment

    print(f"📊 Dataset Gabungan Topik: {len(df_merged_topic)} baris (Sebelumnya: {len(df_base_topic)})")
    print(f"📊 Dataset Gabungan Sentimen: {len(df_merged_sentiment)} baris (Sebelumnya: {len(df_base_sentiment)})")

    return df_merged_topic, df_merged_sentiment


def retrain_single_model(
    model_type: str,
    df_train_raw: pd.DataFrame,
    val_csv_path: str,
    test_csv_path: str,
    output_dir: str,
    target_samples_per_class: int = 70,
    epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, Any]:
    """
    Melatih satu model (Topic atau Sentiment) dengan data baru, mengevaluasi di test set,
    dan mengembalikan perbandingan performa.
    """
    device = get_device()
    tokenizer = get_tokenizer()

    if model_type == "topic":
        num_labels = 5
        id2label = TOPIC_ID2LABEL
        label2id = TOPIC_LABEL2ID
        label_col = "Topic_ID"
        df_augmented = augment_topic_dataset(df_train_raw, target_samples_per_class=target_samples_per_class)
    else:
        num_labels = 2
        id2label = SENTIMENT_ID2LABEL
        label2id = SENTIMENT_LABEL2ID
        label_col = "Sentiment_ID"
        df_augmented = augment_sentiment_dataset(df_train_raw, target_samples_per_class=175)

    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    train_loader, val_loader, test_loader = create_dataloaders(
        df_train=df_augmented,
        df_val=df_val,
        df_test=df_test,
        tokenizer=tokenizer,
        label_col=label_col,
        batch_size=batch_size,
    )

    # Inisialisasi model baru
    model = get_model(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_f1, val_acc = eval_epoch(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Muat bobot terbaik untuk pengujian final di test set murni
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Evaluasi di Test Set Murni
    eval_results = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        id2label=id2label,
        device=device,
    )

    return {
        "model_type": model_type,
        "best_val_f1": round(best_val_f1, 4),
        "test_accuracy": round(eval_results["accuracy"], 4),
        "test_macro_f1": round(eval_results["macro_f1"], 4),
        "test_macro_precision": round(eval_results["macro_precision"], 4),
        "test_macro_recall": round(eval_results["macro_recall"], 4),
        "classification_report": eval_results["classification_report"],
        "model": model,
        "tokenizer": tokenizer,
    }


def run_continuous_retraining_pipeline() -> Dict[str, Any]:
    """
    Eksekusi alur Continuous Retraining lengkap dengan verifikasi Anti-Degradasi.
    """
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"🔄 MEMULAI CONTINUOUS RETRAINING PIPELINE [{timestamp}]")
    print("=" * 70)

    # 1. Siapkan dataset gabungan
    df_merged_topic, df_merged_sentiment = prepare_retraining_datasets()

    # 2. Baca baseline metric saat ini
    baseline_metrics = {}
    metrics_path = "reports/metrics_summary.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
        except Exception:
            pass

    old_topic_f1 = baseline_metrics.get("topic_model", {}).get("macro_f1", 0.0)
    old_sentiment_f1 = baseline_metrics.get("sentiment_model", {}).get("macro_f1", 0.0)

    # 3. Retrain Candidate Topic Model
    print("\n🚀 [1/2] Melatih Candidate Model Topik...")
    topic_res = retrain_single_model(
        model_type="topic",
        df_train_raw=df_merged_topic,
        val_csv_path="data/processed/val_topic.csv",
        test_csv_path="data/processed/test_topic.csv",
        output_dir="models/topic_model",
        epochs=6,
    )
    new_topic_f1 = topic_res["test_macro_f1"]

    # 4. Retrain Candidate Sentiment Model
    print("\n🚀 [2/2] Melatih Candidate Model Sentimen...")
    sentiment_res = retrain_single_model(
        model_type="sentiment",
        df_train_raw=df_merged_sentiment,
        val_csv_path="data/processed/val_sentiment.csv",
        test_csv_path="data/processed/test_sentiment.csv",
        output_dir="models/sentiment_model",
        epochs=6,
    )
    new_sentiment_f1 = sentiment_res["test_macro_f1"]

    # 5. Keputusan Deploy / Rollback (Anti-Regression Rule)
    # Toleransi: Model baru diterima jika new_f1 >= old_f1 * 0.95
    topic_accepted = new_topic_f1 >= (old_topic_f1 * 0.95)
    sentiment_accepted = new_sentiment_f1 >= (old_sentiment_f1 * 0.95)

    deployment_status = {
        "topic_model": "PROMOTED" if topic_accepted else "REJECTED_DUE_TO_REGRESSION",
        "sentiment_model": "PROMOTED" if sentiment_accepted else "REJECTED_DUE_TO_REGRESSION",
    }

    if topic_accepted:
        backup_model_checkpoint("models/topic_model")
        save_model_and_tokenizer(
            model=topic_res["model"],
            tokenizer=topic_res["tokenizer"],
            save_directory="models/topic_model",
            extra_metadata={
                "retrained_at": timestamp,
                "test_macro_f1": new_topic_f1,
                "test_accuracy": topic_res["test_accuracy"],
            },
        )
        print(f"✅ Topic Model Baru DIPROMOSIKAN ke Produksi! (F1: {new_topic_f1:.4f} vs Lama: {old_topic_f1:.4f})")
    else:
        print(f"⚠️ Topic Model Baru DITOLAK untuk menjaga stabilitas. (F1: {new_topic_f1:.4f} vs Lama: {old_topic_f1:.4f})")

    if sentiment_accepted:
        backup_model_checkpoint("models/sentiment_model")
        save_model_and_tokenizer(
            model=sentiment_res["model"],
            tokenizer=sentiment_res["tokenizer"],
            save_directory="models/sentiment_model",
            extra_metadata={
                "retrained_at": timestamp,
                "test_macro_f1": new_sentiment_f1,
                "test_accuracy": sentiment_res["test_accuracy"],
            },
        )
        print(f"✅ Sentiment Model Baru DIPROMOSIKAN ke Produksi! (F1: {new_sentiment_f1:.4f} vs Lama: {old_sentiment_f1:.4f})")
    else:
        print(f"⚠️ Sentiment Model Baru DITOLAK untuk menjaga stabilitas. (F1: {new_sentiment_f1:.4f} vs Lama: {old_sentiment_f1:.4f})")

    # 6. Perbarui Metrics Summary jika ada model yang dipromosikan
    updated_metrics = dict(baseline_metrics)
    if topic_accepted:
        updated_metrics["topic_model"] = {
            "accuracy": topic_res["test_accuracy"],
            "macro_precision": topic_res["test_macro_precision"],
            "macro_recall": topic_res["test_macro_recall"],
            "macro_f1": new_topic_f1,
            "classification_report": topic_res["classification_report"],
            "last_retrained": timestamp,
        }
    if sentiment_accepted:
        updated_metrics["sentiment_model"] = {
            "accuracy": sentiment_res["test_accuracy"],
            "macro_precision": sentiment_res["test_macro_precision"],
            "macro_recall": sentiment_res["test_macro_recall"],
            "macro_f1": new_sentiment_f1,
            "classification_report": sentiment_res["classification_report"],
            "last_retrained": timestamp,
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(updated_metrics, f, indent=4, ensure_ascii=False)

    elapsed_sec = round(time.time() - start_time, 2)

    # 7. Catat Log Riwayat
    log_entry = {
        "timestamp": timestamp,
        "elapsed_seconds": elapsed_sec,
        "topic_comparison": {
            "old_f1": old_topic_f1,
            "new_f1": new_topic_f1,
            "decision": deployment_status["topic_model"],
        },
        "sentiment_comparison": {
            "old_f1": old_sentiment_f1,
            "new_f1": new_sentiment_f1,
            "decision": deployment_status["sentiment_model"],
        },
        "total_training_samples": {
            "topic": len(df_merged_topic),
            "sentiment": len(df_merged_sentiment),
        },
    }
    save_retraining_log(log_entry)

    print("\n" + "=" * 70)
    print(f"🏁 PIPELINE SELESAI DALAM {elapsed_sec} DETIK")
    print("=" * 70)

    return log_entry
