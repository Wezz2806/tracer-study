"""
Skrip Eksekusi Otomatis Evaluasi Dual-Model IndoBERT Tracer Study.
Menguji performa Model Topik & Model Sentimen pada Test Set murni tanpa data leakage,
serta menghasilkan visualisasi Confusion Matrix, WordCloud, dan Classification Report.
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

from src.model import (
    load_model_and_tokenizer,
    TOPIC_ID2LABEL,
    SENTIMENT_ID2LABEL,
)
from src.evaluate import evaluate_model, print_evaluation_summary
from src.visualize import (
    plot_confusion_matrix,
    plot_class_performance,
    generate_wordclouds_per_category,
)
from src.train import get_device


def main():
    device = get_device()

    print("\n" + "=" * 80)
    print("📊 PIPELINE EVALUASI DUAL-MODEL INDOBERT TRACER STUDY (CRISP-DM FASE 4)")
    print("=" * 80)

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/wordclouds", exist_ok=True)

    summary_metrics = {}

    # =========================================================================
    # 1. EVALUASI MODEL 1: TOPIK KRITIK & SARAN (5 KELAS)
    # =========================================================================
    topic_model_dir = "models/topic_model"
    test_topic_path = "data/processed/test_topic.csv"

    if os.path.exists(topic_model_dir) and os.path.exists(test_topic_path):
        print("\n🔍 [1/2] Mengevaluasi Model Topik pada Test Set Murni...")
        df_test_topic = pd.read_csv(test_topic_path)
        topic_model, topic_tokenizer, _ = load_model_and_tokenizer(topic_model_dir, device=device)

        results_topic = evaluate_model(
            model=topic_model,
            tokenizer=topic_tokenizer,
            test_df=df_test_topic,
            id2label=TOPIC_ID2LABEL,
            device=device,
        )

        print_evaluation_summary(results_topic, "IndoBERT Topic Classifier (5 Kelas)")

        # Simpan Prediksi Detail
        results_topic["predictions_df"].to_csv("reports/test_topic_predictions.csv", index=False)

        # Plot Confusion Matrix
        plot_confusion_matrix(
            cm=results_topic["confusion_matrix"],
            class_names=results_topic["label_names"],
            title="Confusion Matrix: IndoBERT Topic Classifier (Test Set Murni)",
            save_path="reports/figures/confusion_matrix_topic.png",
        )

        # Plot Class Performance
        plot_class_performance(
            clf_report_dict=results_topic["classification_report_dict"],
            title="Metrik Performa Per-Kelas: Model Topik",
            save_path="reports/figures/class_performance_topic.png",
        )

        summary_metrics["topic_model"] = {
            "accuracy": results_topic["accuracy"],
            "macro_precision": results_topic["macro_precision"],
            "macro_recall": results_topic["macro_recall"],
            "macro_f1": results_topic["macro_f1"],
            "weighted_f1": results_topic["weighted_f1"],
            "classification_report": results_topic["classification_report_dict"],
        }
    else:
        print(f"⚠️ Model Topik ({topic_model_dir}) atau data uji ({test_topic_path}) tidak ditemukan!")

    # =========================================================================
    # 2. EVALUASI MODEL 2: SENTIMEN TESTIMONI (2 KELAS)
    # =========================================================================
    sentiment_model_dir = "models/sentiment_model"
    test_sentiment_path = "data/processed/test_sentiment.csv"

    if os.path.exists(sentiment_model_dir) and os.path.exists(test_sentiment_path):
        print("\n🔍 [2/2] Mengevaluasi Model Sentimen pada Test Set Murni...")
        df_test_sentiment = pd.read_csv(test_sentiment_path)
        sentiment_model, sentiment_tokenizer, _ = load_model_and_tokenizer(sentiment_model_dir, device=device)

        results_sentiment = evaluate_model(
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            test_df=df_test_sentiment,
            id2label=SENTIMENT_ID2LABEL,
            device=device,
        )

        print_evaluation_summary(results_sentiment, "IndoBERT Sentiment Classifier (2 Kelas)")

        # Simpan Prediksi Detail
        results_sentiment["predictions_df"].to_csv("reports/test_sentiment_predictions.csv", index=False)

        # Plot Confusion Matrix
        plot_confusion_matrix(
            cm=results_sentiment["confusion_matrix"],
            class_names=results_sentiment["label_names"],
            title="Confusion Matrix: IndoBERT Sentiment Classifier (Test Set Murni)",
            save_path="reports/figures/confusion_matrix_sentiment.png",
        )

        # Plot Class Performance
        plot_class_performance(
            clf_report_dict=results_sentiment["classification_report_dict"],
            title="Metrik Performa Per-Kelas: Model Sentimen",
            save_path="reports/figures/class_performance_sentiment.png",
        )

        summary_metrics["sentiment_model"] = {
            "accuracy": results_sentiment["accuracy"],
            "macro_precision": results_sentiment["macro_precision"],
            "macro_recall": results_sentiment["macro_recall"],
            "macro_f1": results_sentiment["macro_f1"],
            "weighted_f1": results_sentiment["weighted_f1"],
            "classification_report": results_sentiment["classification_report_dict"],
        }
    else:
        print(f"⚠️ Model Sentimen ({sentiment_model_dir}) atau data uji ({test_sentiment_path}) tidak ditemukan!")

    # =========================================================================
    # 3. GENERASI WORDCLOUD ANALISIS KATA KUNCI
    # =========================================================================
    print("\n☁️ Menghasilkan WordCloud Analisis Kata Kunci...")
    try:
        # Gabungkan data asli murni untuk topik (Train + Val + Test)
        df_topic_all = pd.concat([
            pd.read_csv("data/processed/train_topic.csv"),
            pd.read_csv("data/processed/val_topic.csv"),
            pd.read_csv("data/processed/test_topic.csv"),
        ], ignore_index=True)

        generate_wordclouds_per_category(
            df=df_topic_all,
            text_col="teks_bersih",
            label_col="label",
            id2label=TOPIC_ID2LABEL,
            save_dir="reports/wordclouds/topics",
        )

        # Gabungkan data asli murni untuk sentimen (Train + Val + Test)
        df_sent_all = pd.concat([
            pd.read_csv("data/processed/train_sentiment.csv"),
            pd.read_csv("data/processed/val_sentiment.csv"),
            pd.read_csv("data/processed/test_sentiment.csv"),
        ], ignore_index=True)

        generate_wordclouds_per_category(
            df=df_sent_all,
            text_col="teks_bersih",
            label_col="label",
            id2label=SENTIMENT_ID2LABEL,
            save_dir="reports/wordclouds/sentiments",
        )
    except Exception as e:
        print(f"⚠️ Peringatan saat membuat WordCloud: {e}")

    # =========================================================================
    # 4. SIMPAN REKAP METRIK KE JSON
    # =========================================================================
    metrics_summary_path = "reports/metrics_summary.json"
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🎉 SELURUH PROSES EVALUASI & VISUALISASI BERHASIL!")
    print(f"• Ringkasan Metrik (JSON) : {metrics_summary_path}")
    print(f"• Grafik Confusion Matrix  : reports/figures/")
    print(f"• Gambar WordCloud         : reports/wordclouds/")
    print(f"• Prediksi Detail Uji      : reports/test_*_predictions.csv")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
