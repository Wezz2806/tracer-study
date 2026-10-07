"""
Modul Preprocessing Teks untuk Sistem NLP Tracer Study IndoBERT.
Mengikuti kaidah pemrosesan teks khusus model Transformer (Contextual Language Model):
- Case folding terkendali
- Pembersihan tanda baca berulang dan karakter non-ASCII
- Normalisasi slang, singkatan, dan istilah khas kampus/mahasiswa
- Penyaringan teks non-kontributif (noise)
- TANPA stemming agresif dan TANPA penghapusan stopwords penting (agar struktur semantik & intensitas sentimen tetap utuh).
"""

import re
import unicodedata
from typing import Dict, List, Optional

# Kamus Normalisasi Slang, Singkatan, dan Istilah Kampus
SLANG_DICTIONARY: Dict[str, str] = {
    # Singkatan Umum Bahasa Indonesia
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "utk": "untuk",
    "untk": "untuk",
    "bgt": "sangat",
    "bngt": "sangat",
    "bgtt": "sangat",
    "banget": "sangat",
    "bgtu": "begitu",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "jgn": "jangan",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "sdh": "sudah",
    "udh": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "tp": "tetapi",
    "tpi": "tetapi",
    "krn": "karena",
    "karna": "karena",
    "dlm": "dalam",
    "dr": "dari",
    "dri": "dari",
    "pd": "pada",
    "bkn": "bukan",
    "sy": "saya",
    "km": "kamu",
    "org": "orang",
    "trs": "terus",
    "trus": "terus",
    "bisa": "bisa",
    "bs": "bisa",
    "msh": "masih",
    "cm": "hanya",
    "cuma": "hanya",
    "aja": "saja",
    "doang": "saja",
    "lg": "lagi",
    "bbrp": "beberapa",
    "byk": "banyak",
    "moga": "semoga",
    "smoga": "semoga",
    "kira": "kiranya",
    "makin": "semakin",
    
    # Istilah Akademik, Kampus, dan Karier
    "undika": "universitas dinamika",
    "dinamika": "universitas dinamika",
    "univ": "universitas",
    "prodi": "program studi",
    "jur": "jurusan",
    "mhs": "mahasiswa",
    "matkul": "mata kuliah",
    "makul": "mata kuliah",
    "dosendosen": "dosen-dosen",
    "dosen2": "dosen-dosen",
    "staff": "staf",
    "praktek": "praktik",
    "praktikum": "praktikum",
    "lapang": "lapangan",
    "loker": "lowongan kerja",
    "magang": "magang",
    "internship": "magang",
    "softskill": "keterampilan interpersonal",
    "soft skill": "keterampilan interpersonal",
    "softskills": "keterampilan interpersonal",
    "hardskill": "keterampilan teknis",
    "hard skill": "keterampilan teknis",
    "hardskills": "keterampilan teknis",
    "upgrade": "peningkatan",
    "match": "sesuai",
    "deadline": "tenggat waktu",
    "feedback": "umpan balik",
    "cv": "curriculum vitae",
    "ai": "kecerdasan buatan",
    "ms,access": "microsoft access",
    "ms access": "microsoft access",
    "msaccess": "microsoft access",
    "excel": "microsoft excel",
    "finance": "keuangan",
    "nice": "bagus",
    "komen": "komentar",
    "no komen": "tidak ada komentar",
}

# Pola Teks Non-Kontributif (Noise)
NOISE_PATTERNS: List[str] = [
    r"^[\s\-_.,/]*$",
    r"^(tidak ada|tdk ada|ga ada|gak ada|ngga ada|nihil|belum ada|blm ada)$",
    r"^(no comment|no komen|no komen nice|emm|emm tidak ada|none|null|na|-)$",
    r"^(cukup|sudah cukup|oke|ok|good|sip)$"
]


def is_noise_text(text: Optional[str]) -> bool:
    """
    Mengecek apakah sebuah teks tergolong noise / non-kontributif.
    Contoh: 'tidak ada', '-', 'No komen nice', 'emm tidak ada'.
    """
    if text is None:
        return True
    
    cleaned = text.strip().lower()
    if len(cleaned) == 0:
        return True
        
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, cleaned):
            return True
            
    # Teks terlalu pendek tanpa makna substansial (< 4 karakter selain kata bermakna)
    if len(cleaned) < 4 and cleaned not in ["baik", "puas", "siap"]:
        return True
        
    return False


def normalize_unicode(text: str) -> str:
    """Menghilangkan karakter non-ASCII, aksen, dan emotikon."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_text(text: Optional[str], normalize_slang: bool = True) -> str:
    """
    Membersihkan teks masukan sesuai standar BERT IndoBERT:
    1. Konversi ke lowercase (Case Folding)
    2. Normalisasi Unicode & pembersihan URL/HTML
    3. Perapian tanda baca berulang
    4. Normalisasi singkatan & slang menggunakan kamus terstruktur
    5. Normalisasi spasi berlebih
    """
    if text is None:
        return ""
        
    # 1. Konversi string & Lowercase
    text = str(text).strip().lower()
    if not text:
        return ""
        
    # 2. Unicode Normalization
    text = normalize_unicode(text)
    
    # 3. Hapus URL dan tag HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    
    # 4. Rapikan tanda baca berulang (misal ??? -> ?, !!! -> !, ... -> .)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r",+", ",", text)
    
    # 5. Hilangkan karakter non-alfanumerik selain tanda baca umum (. , ? ! - )
    text = re.sub(r"[^\w\s.,?!-]", " ", text)
    
    # 6. Normalisasi Slang & Singkatan
    if normalize_slang:
        # Pisahkan token dengan regex kata utuh
        tokens = re.findall(r"\b[\w-]+\b|[.,?!]", text)
        normalized_tokens = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in SLANG_DICTIONARY:
                normalized_tokens.append(SLANG_DICTIONARY[token_lower])
            else:
                normalized_tokens.append(token)
        text = " ".join(normalized_tokens)
        
        # Perbaiki spasi sebelum tanda baca
        text = re.sub(r"\s+([.,?!])", r"\1", text)
        
    # 7. Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


if __name__ == "__main__":
    # Pengujian mandiri (Self-test)
    sample_texts = [
        "Perlu peningkatan fasilitas lab komputer, loker magang diperbanyak bgt!!",
        "No komen nice",
        "emm tidak ada",
        "Dosen2 di undika sangat membantu dlm penyusunan tugas akhir & hardskill.",
        "Materi kuliah AI & ms,access hrs ditingkatkan biar match dgn dunia kerja."
    ]
    
    print("=== PENGUJIAN MODUL PREPROCESSING ===")
    for s in sample_texts:
        noise = is_noise_text(s)
        cleaned = clean_text(s)
        print(f"Asli    : {s}")
        print(f"Noise?  : {noise}")
        print(f"Bersih  : {cleaned}")
        print("-" * 50)
