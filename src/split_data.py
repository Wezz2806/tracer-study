"""
Modul Pembagian Data Terstratifikasi (Stratified Splitting)
Membagi dataset asli bersih menjadi Train (70%), Val (15%), dan Test (15%)
dengan menjaga proporsi kelas label (Topic & Sentiment).

PENTING:
Data Val dan Test disimpan murni tanpa augmentasi untuk evaluasi objektif.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import clean_text, is_noise_text


# Pemetaan Standar 5 Topik Tracer Study
TOPIC_LABEL_MAP: Dict[int, str] = {
    0: "Kurikulum & Pembelajaran",
    1: "Fasilitas & Sarana Prasarana",
    2: "Dosen & Tenaga Pengajar",
    3: "Karier, Magang & Kesiapan Kerja",
    4: "Layanan Kemahasiswaan & Administrasi"
}

# Pemetaan Sentimen Biner
SENTIMENT_LABEL_MAP: Dict[int, str] = {
    0: "Negatif",
    1: "Positif"
}


def map_topic_from_cluster_and_keywords(row: pd.Series) -> int:
    """
    Memetakan masukan kritik/saran ke 5 topik standar berdasarkan
    analisis semantik teks bersih dan cluster id asal.
    """
    text = str(row.get("teks_bersih", "")).lower()
    
    # Deteksi Topik 3: Karier, Magang & Kesiapan Kerja
    if any(k in text for k in ["loker", "lowongan kerja", "magang", "dunia kerja", "kerja sama industri", "industri", "karier", "perusahaan", "relasi antar sesama pengusaha", "finance"]):
        return 3
    # Deteksi Topik 1: Fasilitas & Sarana Prasarana
    elif any(k in text for k in ["fasilitas", "lab", "laboratorium", "komputer", "gedung", "ruang", "alat", "sarana", "studio", "ac", "wifi"]):
        return 1
    # Deteksi Topik 2: Dosen & Tenaga Pengajar
    elif any(k in text for k in ["dosen", "dosen-dosen", "ajar", "pengajar", "bimbing", "dosen pembimbing", "staf", "staff"]):
        return 2
    # Deteksi Topik 4: Layanan Kemahasiswaan & Administrasi
    elif any(k in text for k in ["layanan", "pelayanan", "administrasi", "birokrasi", "kegiatan", "hima", "bem", "biaya", "uang kuliah"]):
        return 4
    # Default ke Topik 0: Kurikulum & Pembelajaran (materi, teori, praktik, teknologi, kurikulum, skill)
    else:
        return 0


def map_sentiment_from_testimony(row: pd.Series) -> int:
    """
    Memetakan testimoni ke sentimen biner (0: Negatif, 1: Positif).
    Testimoni alumni pada umumnya bernada apresiatif (Positif),
    kecuali terdapat kalimat ketidakpuasan/keluhan spesifik.
    """
    text = str(row.get("teks_bersih", "")).lower()
    
    # Indikator Negatif / Keluhan dalam testimoni
    negative_indicators = ["kurang", "kecewa", "buruk", "tidak sesuai", "sayang sekali", "susah", "sulit", "jelek", "mengecewakan"]
    if any(k in text for k in negative_indicators) and not any(p in text for p in ["tidak mengecewakan", "tidak buruk"]):
        return 0
    return 1


def prepare_topic_dataset(raw_combined_csv: str) -> pd.DataFrame:
    """Menyiapkan dataset topik dari gabungan Kritik dan Saran non-noise."""
    df = pd.read_csv(raw_combined_csv)
    # Filter hanya Kritik dan Saran
    df_topic = df[df["kategori_sumber"].isin(["Kritik", "Saran"])].copy()
    
    # Buat label topik terstandar
    df_topic["label"] = df_topic.apply(map_topic_from_cluster_and_keywords, axis=1)
    df_topic["label_name"] = df_topic["label"].map(TOPIC_LABEL_MAP)
    
    # Pilih kolom esensial
    df_topic = df_topic[["nim", "program_studi", "kategori_sumber", "teks_asli", "teks_bersih", "label", "label_name"]].reset_index(drop=True)
    return df_topic


def prepare_sentiment_dataset(raw_combined_csv: str) -> pd.DataFrame:
    """Menyiapkan dataset sentimen dari data Testimoni (+ Kritik berbobot sentimen)."""
    df = pd.read_csv(raw_combined_csv)
    
    records = []
    for _, row in df.iterrows():
        source = row["kategori_sumber"]
        text_clean = str(row["teks_bersih"])
        text_raw = str(row["teks_asli"])
        
        if source == "Testimoni":
            label = map_sentiment_from_testimony(row)
            records.append({
                "nim": row["nim"],
                "program_studi": row["program_studi"],
                "kategori_sumber": source,
                "teks_asli": text_raw,
                "teks_bersih": text_clean,
                "label": label,
                "label_name": SENTIMENT_LABEL_MAP[label]
            })
        elif source == "Kritik":
            # Kritik yang menyoroti kekurangan otomatis bernilai sentimen Negatif (0)
            records.append({
                "nim": row["nim"],
                "program_studi": row["program_studi"],
                "kategori_sumber": source,
                "teks_asli": text_raw,
                "teks_bersih": text_clean,
                "label": 0,
                "label_name": SENTIMENT_LABEL_MAP[0]
            })
            
    df_sentiment = pd.DataFrame(records)
    return df_sentiment


def split_and_save_dataset(df: pd.DataFrame, task_name: str, processed_dir: str):
    """
    Melakukan Stratified Split 70% Train, 15% Val, 15% Test
    dan menyimpan file CSV ke direktori data/processed/.
    """
    os.makedirs(processed_dir, exist_ok=True)
    label_map = TOPIC_LABEL_MAP if task_name == "topic" else SENTIMENT_LABEL_MAP
    
    # Cek distribusi kelas
    class_counts = df["label"].value_counts().sort_index()
    print(f"\n--- Distribusi Kelas Asli [{task_name.upper()}] ---")
    for lbl, cnt in class_counts.items():
        name = label_map.get(lbl, str(lbl))
        print(f"  • Label {lbl} ({name}): {cnt} sampel")
        
    # Split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"] if min(class_counts) > 1 else None
    )
    
    # Split Temp (30%) menjadi 15% Val dan 15% Test (50:50 dari Temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"] if min(temp_df["label"].value_counts()) > 1 else None
    )
    
    # Simpan ke CSV
    train_path = os.path.join(processed_dir, f"train_{task_name}.csv")
    val_path = os.path.join(processed_dir, f"val_{task_name}.csv")
    test_path = os.path.join(processed_dir, f"test_{task_name}.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ [{task_name.capitalize()}] Berhasil Displit:")
    print(f"   • Train Set (70%) : {len(train_df)} sampel -> {train_path}")
    print(f"   • Val Set   (15%) : {len(val_df)} sampel -> {val_path}")
    print(f"   • Test Set  (15%) : {len(test_df)} sampel -> {test_path}")


def run_splitting():
    raw_combined = os.path.join(parent_dir, "data", "raw", "combined_tracer_study.csv")
    if not os.path.exists(raw_combined):
        print(f"File {raw_combined} belum ada. Menjalankan scripts/eda_summary.py terlebih dahulu...")
        from scripts.eda_summary import run_eda
        run_eda()
        
    processed_dir = os.path.join(parent_dir, "data", "processed")
    
    print("=" * 70)
    print("✂️ PEMBAGIAN DATASET TERSTRATIFIKASI (STRATIFIED DATA SPLIT)")
    print("=" * 70)
    
    # 1. Modul Topik
    df_topic = prepare_topic_dataset(raw_combined)
    split_and_save_dataset(df_topic, "topic", processed_dir)
    
    # 2. Modul Sentimen
    df_sentiment = prepare_sentiment_dataset(raw_combined)
    split_and_save_dataset(df_sentiment, "sentiment", processed_dir)
    
    print("\n" + "=" * 70)
    print("🎉 Seluruh dataset berhasil displit dan siap untuk tahap Augmentasi!")
    print("=" * 70)


if __name__ == "__main__":
    run_splitting()
