"""
Modul Visualisasi Hasil Evaluasi dan Analisis Teks untuk Sistem NLP Tracer Study.
Menghasilkan grafik Training Loss/F1 Kurva, Confusion Matrix Heatmap, Metrik Per-Kelas,
dan WordCloud Analisis Kata Kunci per Topik / Sentimen.
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Daftar Stopwords Bahasa Indonesia untuk WordCloud
INDONESIAN_STOPWORDS = set([
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
    "ia", "seperti", "jika", "jika", "sehingga", "kembali", "dan", "ini", "karena",
    "oleh", "saat", "harus", "sementara", "setelah", "belum", "kami", "sekitar",
    "mereka", "didapat", "bisa", "bila", "tentang", "demi", "dimana", "dengan",
    "akan", "juga", "dari", "ada", "serta", "sudah", "saya", "kamu", "kita", "nya",
    "tersebut", "dapat", "lebih", "agar", "dalam", "bagi", "banyak", "hal", "lagi",
    "mau", "saja", "tetap", "bukan", "hanya", "masih", "sangat", "tiap", "tempat",
    "undika", "universitas", "dinamika", "kampus", "mahasiswa", "alumni", "kuliah",
])


def set_visual_style() -> None:
    """Mengatur konfigurasi visual Matplotlib & Seaborn agar estetik dan modern."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_training_history(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Membuat grafik Loss Curve dan Validation Macro F1 / Accuracy per epoch.

    Args:
        history: Kamus riwayat pelatihan dari train_indobert_model.
        save_path: Lokasi penyimpanan file gambar PNG.

    Returns:
        Objek matplotlib Figure.
    """
    set_visual_style()
    epochs = history["epochs"]
    task_name = history.get("task_name", "IndoBERT Model")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Loss Curve
    ax1.plot(epochs, history["train_loss"], marker="o", linewidth=2.2, color="#1f77b4", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], marker="s", linewidth=2.2, color="#ff7f0e", linestyle="--", label="Validation Loss")
    ax1.set_title(f"Kurva Loss Pelatihan - {task_name}", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Validation Metrics
    ax2.plot(epochs, [v * 100 for v in history["val_macro_f1"]], marker="^", linewidth=2.2, color="#2ca02c", label="Val Macro F1 (%)")
    ax2.plot(epochs, [v * 100 for v in history["val_accuracy"]], marker="d", linewidth=2.2, color="#9467bd", linestyle=":", label="Val Accuracy (%)")

    # Tandai Best Epoch
    best_epoch = history.get("best_epoch", 1)
    best_f1 = history.get("best_val_macro_f1", 0.0) * 100
    ax2.scatter([best_epoch], [best_f1], color="#d62728", s=100, zorder=5, label=f"Best Checkpoint (Ep {best_epoch}: {best_f1:.1f}%)")

    ax2.set_title(f"Metrik Validasi - {task_name}", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Skor (%)", fontsize=10, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"📊 Grafik Training History disimpan di: {save_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Membuat visualisasi Heatmap Confusion Matrix berstandar publikasi/laporan.

    Args:
        cm: Matriks konfusi 2D (numpy array atau list).
        class_names: Daftar nama kelas sesuai urutan index.
        title: Judul grafik.
        save_path: Lokasi penyimpanan file gambar PNG.

    Returns:
        Objek matplotlib Figure.
    """
    set_visual_style()
    cm_arr = np.array(cm)

    # Hitung persentase per baris (Recall normalisasi)
    cm_sum = cm_arr.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm_arr, cm_sum, out=np.zeros_like(cm_arr, dtype=float), where=cm_sum != 0) * 100

    # Format teks annotasi: 'Jumlah (Persentase%)'
    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot[i, j] = f"{cm_arr[i, j]}\n({cm_perc[i, j]:.1f}%)"

    fig_size = (8, 6.5) if len(class_names) <= 3 else (10, 8)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=1,
        linecolor="#ffffff",
        annot_kws={"size": 10, "fontweight": "bold"},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Prediksi Model (Predicted Label)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Label Sebenarnya (True Ground Truth)", fontsize=11, fontweight="bold", labelpad=10)

    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"📊 Heatmap Confusion Matrix disimpan di: {save_path}")

    return fig


def plot_class_performance(
    clf_report_dict: Dict[str, Any],
    title: str = "Performa Model Per-Kelas",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Membuat grafik Bar Chart komparasi Precision, Recall, dan F1-Score per kelas.
    """
    set_visual_style()

    # Filter hanya entri kelas (abaikan 'accuracy', 'macro avg', 'weighted avg')
    classes = [k for k in clf_report_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    precisions = [clf_report_dict[c]["precision"] * 100 for c in classes]
    recalls = [clf_report_dict[c]["recall"] * 100 for c in classes]
    f1s = [clf_report_dict[c]["f1-score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#2ecc71")
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Skor (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Tambahkan label angka di atas setiap bar
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"📊 Grafik Metrik Per-Kelas disimpan di: {save_path}")

    return fig


def generate_wordclouds_per_category(
    df: pd.DataFrame,
    text_col: str = "teks_bersih",
    label_col: str = "label",
    id2label: Optional[Dict[int, str]] = None,
    save_dir: Optional[str] = "reports/wordclouds",
) -> Dict[str, str]:
    """
    Menghasilkan gambar WordCloud untuk setiap kategori label (Topik / Sentimen).

    Args:
        df: DataFrame teks masukan.
        text_col: Kolom teks bersih.
        label_col: Kolom label numerik.
        id2label: Kamus pemetaan label numerik ke nama.
        save_dir: Direktori penyimpanan file WordCloud PNG.

    Returns:
        Kamus nama label -> path gambar yang dihasilkan.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    generated_paths = {}
    unique_labels = sorted(df[label_col].unique())

    # Palet warna WordCloud
    colormaps = ["Blues", "Greens", "Oranges", "Purples", "RdPu", "YlGnBu", "magma"]

    for idx, label_val in enumerate(unique_labels):
        label_name = id2label[label_val] if id2label and label_val in id2label else f"Kategori_{label_val}"
        subset_texts = df[df[label_col] == label_val][text_col].dropna().tolist()
        combined_text = " ".join(subset_texts)

        if not combined_text.strip():
            continue

        cmap = colormaps[idx % len(colormaps)]

        wc = WordCloud(
            width=800,
            height=450,
            background_color="white",
            stopwords=INDONESIAN_STOPWORDS,
            colormap=cmap,
            max_words=80,
            contour_width=1,
            contour_color="#e0e0e0",
            random_state=42,
        ).generate(combined_text)

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"WordCloud: {label_name}", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout(pad=0)

        # Simpan file
        sanitized_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label_name.lower())
        file_path = os.path.join(save_dir, f"wordcloud_{sanitized_name}.png")
        fig.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        generated_paths[label_name] = file_path
        print(f"☁️ WordCloud [{label_name}] disimpan di: {file_path}")

    return generated_paths
