"""
Modul Model Architecture & Factory untuk Dual-Model IndoBERT Tracer Study.
Menyediakan konfigurasi IndoBERT Multi-Class Topic Classifier dan Binary Sentiment Classifier.
"""

import os
import json
from typing import Dict, Tuple, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

# Model Backbone Pretrained
PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"

# Pemetaan Label Topik (5 Kelas)
TOPIC_ID2LABEL: Dict[int, str] = {
    0: "Kurikulum & Pembelajaran",
    1: "Fasilitas & Sarana Prasarana",
    2: "Dosen & Tenaga Pengajar",
    3: "Karier, Magang & Kesiapan Kerja",
    4: "Layanan Kemahasiswaan & Administrasi Kampus",
}
TOPIC_LABEL2ID: Dict[str, int] = {v: k for k, v in TOPIC_ID2LABEL.items()}

# Pemetaan Label Sentimen (2 Kelas)
SENTIMENT_ID2LABEL: Dict[int, str] = {
    0: "Negatif",
    1: "Positif",
}
SENTIMENT_LABEL2ID: Dict[str, int] = {v: k for k, v in SENTIMENT_ID2LABEL.items()}


def get_tokenizer(pretrained_name: str = PRETRAINED_MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Memuat tokenizer pre-trained IndoBERT.

    Args:
        pretrained_name: Nama repositori model Hugging Face atau path lokal.

    Returns:
        AutoTokenizer siap pakai.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    return tokenizer


def get_model(
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    pretrained_name: str = PRETRAINED_MODEL_NAME,
    classifier_dropout: float = 0.2,
) -> PreTrainedModel:
    """
    Membangun arsitektur model IndoBERT Sequence Classification dengan classification head kustom.

    Args:
        num_labels: Jumlah kelas keluaran.
        id2label: Kamus pemetaan ID numerik ke nama label.
        label2id: Kamus pemetaan nama label ke ID numerik.
        pretrained_name: Nama model dasar pre-trained.
        classifier_dropout: Nilai dropout pada classification head.

    Returns:
        Model PyTorch AutoModelForSequenceClassification.
    """
    # Konversi key id2label ke string untuk kompatibilitas config JSON Hugging Face
    id2label_str = {str(k): v for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        id2label=id2label_str,
        label2id=label2id,
        classifier_dropout=classifier_dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=config,
    )

    return model


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    save_directory: str,
    extra_metadata: Optional[Dict] = None,
) -> None:
    """
    Menyimpan model, tokenizer, dan metadata pelatihan ke direktori target.

    Args:
        model: Objek model Hugging Face PyTorch.
        tokenizer: Objek tokenizer.
        save_directory: Lokasi folder penyimpanan.
        extra_metadata: Kamus metadata tambahan (misal: metrik evaluasi terbaik).
    """
    os.makedirs(save_directory, exist_ok=True)

    # Simpan bobot model & konfigurasi
    model.save_pretrained(save_directory)

    # Simpan tokenizer
    tokenizer.save_pretrained(save_directory)

    # Simpan metadata tambahan jika ada
    if extra_metadata:
        metadata_path = os.path.join(save_directory, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(extra_metadata, f, indent=4, ensure_ascii=False)

    print(f"✅ Model dan Tokenizer berhasil disimpan di: {save_directory}")


def load_model_and_tokenizer(
    model_directory: str,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict]:
    """
    Memuat model dan tokenizer dari folder lokal yang telah disimpan.

    Args:
        model_directory: Lokasi folder penyimpanan model.
        device: Device target (CPU / CUDA).

    Returns:
        Tuple (model, tokenizer, metadata).
    """
    if not os.path.exists(model_directory):
        raise FileNotFoundError(f"Direktori model tidak ditemukan: {model_directory}")

    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)

    if device is not None:
        model.to(device)

    model.eval()

    # Muat metadata jika tersedia
    metadata = {}
    metadata_path = os.path.join(model_directory, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, tokenizer, metadata
