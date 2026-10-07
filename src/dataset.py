"""
Modul Dataset & DataLoader untuk Sistem NLP Tracer Study IndoBERT.
Mengelola konversi teks masukan menjadi tensor PyTorch siap latih dengan IndoBERT Tokenizer.
"""

from typing import List, Optional, Union, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizerBase


class TracerStudyDataset(Dataset):
    """
    Dataset PyTorch untuk data masukan Tracer Study (Kritik, Saran, Testimoni).
    Mendukung tokenisasi dinamis dengan penanganan padding dan truncation.
    """

    def __init__(
        self,
        texts: Union[List[str], pd.Series],
        labels: Optional[Union[List[int], pd.Series]] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        max_length: int = 128,
    ):
        """
        Inisialisasi dataset.

        Args:
            texts: List atau Series teks masukan (teks_bersih).
            labels: List atau Series label numerik ground-truth (opsional untuk inferensi).
            tokenizer: Objek Hugging Face Tokenizer (IndoBERT).
            max_length: Batas panjang maksimum tokenisasi (default: 128).
        """
        if isinstance(texts, pd.Series):
            self.texts = texts.fillna("").astype(str).tolist()
        else:
            self.texts = [str(t) if t is not None else "" for t in texts]

        if labels is not None:
            if isinstance(labels, pd.Series):
                self.labels = labels.astype(int).tolist()
            else:
                self.labels = [int(l) for l in labels]
        else:
            self.labels = None

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
                "text": text,
            }
        else:
            item = {"text": text}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def create_data_loader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    text_col: str = "teks_bersih",
    label_col: Optional[str] = "label",
    batch_size: int = 16,
    shuffle: bool = True,
    max_length: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """
    Helper untuk membuat DataLoader PyTorch dari DataFrame pandas.

    Args:
        df: DataFrame berisi data teks dan label.
        tokenizer: Hugging Face Tokenizer.
        text_col: Nama kolom teks.
        label_col: Nama kolom label (None jika untuk inferensi tanpa label).
        batch_size: Ukuran batch per iterasi.
        shuffle: Apakah data diacak per epoch.
        max_length: Panjang maksimum token.
        num_workers: Jumlah worker multiprocessing untuk data loading.

    Returns:
        DataLoader PyTorch siap pakai.
    """
    texts = df[text_col].tolist()
    labels = df[label_col].tolist() if label_col and label_col in df.columns else None

    dataset = TracerStudyDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
