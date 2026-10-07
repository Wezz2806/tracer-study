"""
Modul Inferensi Terpadu (Dual-Model Predictor) untuk Sistem NLP Tracer Study.
Menerima teks masukan kritik/saran/testimoni alumni, menjalankan text cleaning otomatis,
dan mengembalikan klasifikasi topik, analisis sentimen, serta skor keyakinan (confidence score).
"""

import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import torch
import torch.nn.functional as F

from src.preprocessing import clean_text
from src.model import load_model_and_tokenizer, TOPIC_ID2LABEL, SENTIMENT_ID2LABEL
from src.train import get_device


class TracerStudyPredictor:
    """
    Pipeline Inferensi Dual-Model IndoBERT Tracer Study.
    Menggabungkan Model 1 (Klasifikasi 5 Topik) dan Model 2 (Analisis 2 Sentimen).
    """

    def __init__(
        self,
        topic_model_dir: str = "models/topic_model",
        sentiment_model_dir: str = "models/sentiment_model",
        device: Optional[torch.device] = None,
        max_length: int = 128,
        active_learning_threshold: float = 0.70,
    ):
        """
        Inisialisasi pipeline inferensi.

        Args:
            topic_model_dir: Path direktori checkpoint Model Topik.
            sentiment_model_dir: Path direktori checkpoint Model Sentimen.
            device: Device target komputasi (CPU/GPU).
            max_length: Batas panjang token IndoBERT.
            active_learning_threshold: Ambang batas confidence score untuk flag review manual.
        """
        self.device = device if device is not None else get_device()
        self.max_length = max_length
        self.threshold = active_learning_threshold

        print(f"🔄 Memuat Dual-Model IndoBERT ke {self.device}...")

        # 1. Muat Model Topik jika tersedia
        self.topic_model = None
        self.topic_tokenizer = None
        if os.path.exists(topic_model_dir):
            self.topic_model, self.topic_tokenizer, _ = load_model_and_tokenizer(
                topic_model_dir, device=self.device
            )
            print(f"✅ Model Topik berhasil dimuat dari: {topic_model_dir}")
        else:
            print(f"⚠️ Model Topik belum ditemukan di: {topic_model_dir}")

        # 2. Muat Model Sentimen jika tersedia
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        if os.path.exists(sentiment_model_dir):
            self.sentiment_model, self.sentiment_tokenizer, _ = load_model_and_tokenizer(
                sentiment_model_dir, device=self.device
            )
            print(f"✅ Model Sentimen berhasil dimuat dari: {sentiment_model_dir}")
        else:
            print(f"⚠️ Model Sentimen belum ditemukan di: {sentiment_model_dir}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Melakukan inferensi lengkap pada satu teks masukan.

        Args:
            text: Teks mentah masukan kuesioner Tracer Study.

        Returns:
            Kamus hasil prediksi terstruktur.
        """
        # Preprocessing teks
        cleaned_text = clean_text(text)

        result: Dict[str, Any] = {
            "teks_asli": text,
            "teks_bersih": cleaned_text,
            "topic": None,
            "sentiment": None,
            "requires_human_review": False,
        }

        # 1. Inferensi Model Topik
        if self.topic_model is not None and self.topic_tokenizer is not None:
            encoding = self.topic_tokenizer(
                cleaned_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.topic_model(**encoding)
                probs = F.softmax(outputs.logits, dim=1).squeeze(0)
                confidence, pred_id = torch.max(probs, dim=0)

                pred_id_int = int(pred_id.item())
                conf_val = float(confidence.item())

                # Ambil semua probabilitas per kelas
                probs_dict = {
                    TOPIC_ID2LABEL.get(i, f"Kelas {i}"): round(float(p.item()), 4)
                    for i, p in enumerate(probs)
                }

                result["topic"] = {
                    "label_id": pred_id_int,
                    "label_name": TOPIC_ID2LABEL.get(pred_id_int, "Unknown"),
                    "confidence": round(conf_val, 4),
                    "confidence_percentage": f"{conf_val * 100:.2f}%",
                    "probabilities": probs_dict,
                }

                if conf_val < self.threshold:
                    result["requires_human_review"] = True

        # 2. Inferensi Model Sentimen
        if self.sentiment_model is not None and self.sentiment_tokenizer is not None:
            encoding = self.sentiment_tokenizer(
                cleaned_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sentiment_model(**encoding)
                probs = F.softmax(outputs.logits, dim=1).squeeze(0)
                confidence, pred_id = torch.max(probs, dim=0)

                pred_id_int = int(pred_id.item())
                conf_val = float(confidence.item())

                probs_dict = {
                    SENTIMENT_ID2LABEL.get(i, f"Kelas {i}"): round(float(p.item()), 4)
                    for i, p in enumerate(probs)
                }

                result["sentiment"] = {
                    "label_id": pred_id_int,
                    "label_name": SENTIMENT_ID2LABEL.get(pred_id_int, "Unknown"),
                    "confidence": round(conf_val, 4),
                    "confidence_percentage": f"{conf_val * 100:.2f}%",
                    "probabilities": probs_dict,
                }

                if conf_val < self.threshold:
                    result["requires_human_review"] = True

        return result

    def predict_batch(
        self, texts: Union[List[str], pd.Series]
    ) -> List[Dict[str, Any]]:
        """
        Menjalankan inferensi untuk kumpulan teks (batch).
        """
        results = []
        for t in texts:
            results.append(self.predict(t))
        return results
