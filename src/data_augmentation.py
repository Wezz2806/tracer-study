"""
Modul Data Augmentation Terarah untuk Sistem NLP Tracer Study IndoBERT.
Mengimplementasikan:
1. Back-Translation (ID -> EN -> ID) menggunakan Deep-Translator
2. Contextual Synonym Replacement & Paraphrasing berbasis Kamus Semantik Pendidikan
3. Class Balancing Engine untuk melipatgandakan Data Training (Train Set) menjadi >= 300 data latih seimbang.

CATATAN:
Modul ini HANYA boleh diterapkan pada file train_*.csv (TIDAK pada val atau test).
"""

import os
import sys
import time
import random
import pandas as pd
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import clean_text

# Coba import deep_translator jika terpasang
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


# Kamus Sinonim Kontekstual Akademik & Tracer Study
SYNONYM_MAP: Dict[str, List[str]] = {
    "peningkatan": ["pengembangan", "pembaruan", "penyempurnaan", "kemajuan"],
    "tingkatkan": ["kembangkan", "perbaiki", "perluas", "maksimalkan"],
    "tingkat": ["kembang", "perbaiki", "maksimalkan"],
    "fasilitas": ["sarana dan prasarana", "sarana penunjang", "fasilitas pendukung"],
    "pelatihan": ["program training", "workshop", "pembekalan teknis", "kursus"],
    "latih": ["training", "bimbing", "bekali"],
    "praktik": ["praktikum", "praktek lapangan", "penerapan langsung"],
    "magang": ["internship", "kerja praktik", "program magang industri"],
    "dunia kerja": ["dunia industri", "lingkungan kerja profesional", "lapangan pekerjaan"],
    "bekerja": ["berkarier", "terjun ke industri", "memasuki dunia kerja"],
    "dosen": ["pengajar", "dosen pengampu", "tenaga pendidik"],
    "pembelajaran": ["proses perkuliahan", "kegiatan belajar mengajar", "edukasi"],
    "ilmu": ["pengetahuan", "wawasan", "kompetensi"],
    "sangat": ["amat", "benar-benar", "sungguh"],
    "baik": ["bagus", "berkualitas", "memuaskan"],
    "banyak": ["beragam", "berbagai macam", "sejumlah"],
    "membantu": ["menolong", "mendukung", "mempermudah"],
    "rekan": ["teman", "rekan mahasiswa", "kolega"],
    "kesiapan": ["kesiapan mental dan teknis", "kapabilitas", "kesiapan diri"],
    "kurang": ["masih minim", "perlu ditambah", "belum optimal"],
    "kurangnya": ["masih minimnya", "terbatasnya", "belum optimalnya"]
}


def back_translate(text: str, sleep_time: float = 0.3) -> Optional[str]:
    """
    Melakukan penerjemahan bolak-balik (ID -> EN -> ID)
    untuk menghasilkan variasi parafrase alami.
    """
    if not HAS_TRANSLATOR:
        return None
        
    try:
        # ID -> EN
        en_text = GoogleTranslator(source="id", target="en").translate(text)
        time.sleep(sleep_time)
        # EN -> ID
        id_paraphrased = GoogleTranslator(source="en", target="id").translate(en_text)
        time.sleep(sleep_time)
        
        cleaned = clean_text(id_paraphrased)
        if cleaned and cleaned.lower() != text.lower():
            return cleaned
        return id_paraphrased
    except Exception as e:
        # Fallback jika jaringan lambat / offline
        return None


def contextual_paraphrase(text: str, max_replacements: int = 3) -> str:
    """
    Menghasilkan parafrase teks dengan mengganti kata-kata non-kritis
    menggunakan sinonim kontekstual.
    """
    words = text.split()
    if not words:
        return text
        
    replaced_count = 0
    new_words = []
    
    for word in words:
        clean_w = word.strip(".,?!")
        punct = word[len(clean_w):] if len(word) > len(clean_w) else ""
        
        if clean_w in SYNONYM_MAP and replaced_count < max_replacements:
            if random.random() < 0.6: # Peluang 60% diganti
                synonym = random.choice(SYNONYM_MAP[clean_w])
                new_words.append(synonym + punct)
                replaced_count += 1
                continue
                
        new_words.append(word)
        
    paraphrased = " ".join(new_words)
    return paraphrased if paraphrased != text else text


def generate_augmented_variants(text: str, num_variants: int = 3) -> List[str]:
    """
    Menghasilkan sejumlah variasi teks sintetis yang mempertahankan makna label.
    """
    variants = set()
    cleaned_orig = clean_text(text)
    
    # 1. Coba Back-Translation
    bt_res = back_translate(cleaned_orig)
    if bt_res and bt_res.strip().lower() != cleaned_orig.lower():
        variants.add(bt_res)
        
    # 2. Coba Contextual Paraphrase
    attempts = 0
    while len(variants) < num_variants and attempts < 10:
        attempts += 1
        para = contextual_paraphrase(cleaned_orig, max_replacements=random.randint(1, 4))
        if para.strip().lower() != cleaned_orig.lower():
            variants.add(para)
            
    return list(variants)[:num_variants]


def balance_and_augment_dataset(
    train_csv_path: str,
    output_csv_path: str,
    target_samples_per_class: int,
    task_name: str
) -> pd.DataFrame:
    """
    Membaca dataset latih, menduplikasi & memparafrase kelas minoritas
    hingga setiap kelas mencapai target_samples_per_class.
    """
    print(f"\n🚀 Memulai Data Augmentation untuk [{task_name.upper()}]...")
    df_train = pd.read_csv(train_csv_path)
    
    augmented_rows = []
    
    # Masukkan seluruh data asli terlebih dahulu
    for _, row in df_train.iterrows():
        row_dict = row.to_dict()
        row_dict["is_augmented"] = 0
        augmented_rows.append(row_dict)
        
    # Periksa distribusi per kelas
    class_groups = df_train.groupby("label")
    
    for label, group in class_groups:
        current_count = len(group)
        needed = target_samples_per_class - current_count
        label_name = group.iloc[0]["label_name"] if "label_name" in group.columns else str(label)
        
        print(f"  • Kelas {label} ({label_name}): Data Asli = {current_count}, Target = {target_samples_per_class} (Perlu +{max(0, needed)} sampel)")
        
        if needed <= 0:
            continue
            
        # Generate data sintetis
        generated_for_class = 0
        pool_texts = group[["nim", "program_studi", "kategori_sumber", "teks_asli", "teks_bersih"]].to_dict(orient="records")
        
        while generated_for_class < needed:
            sample_source = random.choice(pool_texts)
            base_text = sample_source["teks_bersih"]
            
            variants = generate_augmented_variants(base_text, num_variants=2)
            if not variants:
                # Fallback parafrase paksa jika sinonim tidak berubah
                variants = [contextual_paraphrase(base_text, max_replacements=2)]
                
            for v in variants:
                if generated_for_class >= needed:
                    break
                new_row = {
                    "nim": f"{sample_source['nim']}_aug",
                    "program_studi": sample_source["program_studi"],
                    "kategori_sumber": sample_source["kategori_sumber"],
                    "teks_asli": sample_source["teks_asli"],
                    "teks_bersih": v,
                    "label": label,
                    "label_name": label_name,
                    "is_augmented": 1
                }
                augmented_rows.append(new_row)
                generated_for_class += 1
                
    df_augmented = pd.DataFrame(augmented_rows)
    # Acak urutan agar tidak terkelompok
    df_augmented = df_augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_augmented.to_csv(output_csv_path, index=False)
    
    print(f"✅ Selesai Augmentasi [{task_name.upper()}]:")
    print(f"   • Total Data Latih Setelah Augmentasi : {len(df_augmented)} baris")
    print(f"   • Distribusi Kelas Baru              : {dict(df_augmented['label'].value_counts())}")
    print(f"   💾 Tersimpan ke                      : {output_csv_path}")
    
    return df_augmented


def run_augmentation():
    processed_dir = os.path.join(parent_dir, "data", "processed")
    augmented_dir = os.path.join(parent_dir, "data", "augmented")
    
    train_topic_csv = os.path.join(processed_dir, "train_topic.csv")
    train_sentiment_csv = os.path.join(processed_dir, "train_sentiment.csv")
    
    if not os.path.exists(train_topic_csv) or not os.path.exists(train_sentiment_csv):
        print("Data train belum displit. Menjalankan src/split_data.py...")
        from src.split_data import run_splitting
        run_splitting()
        
    print("=" * 70)
    print("✨ ENGINE AUGMENTASI DATA NLP TRACER STUDY (CRISP-DM: DATA PREPARATION)")
    print("=" * 70)
    
    # Target untuk Topic (5 kelas x 70 sampel = 350 data latih seimbang)
    out_topic = os.path.join(augmented_dir, "train_topic_augmented.csv")
    balance_and_augment_dataset(train_topic_csv, out_topic, target_samples_per_class=70, task_name="topic")
    
    # Target untuk Sentiment (2 kelas x 175 sampel = 350 data latih seimbang)
    out_sentiment = os.path.join(augmented_dir, "train_sentiment_augmented.csv")
    balance_and_augment_dataset(train_sentiment_csv, out_sentiment, target_samples_per_class=175, task_name="sentiment")
    
    print("\n" + "=" * 70)
    print("🎉 Seluruh Data Training berhasil diperbanyak dan seimbang (>= 350 sampel per tugas)!")
    print("=" * 70)


if __name__ == "__main__":
    run_augmentation()
