"""
REST API Serving Layer berbasis FastAPI untuk Sistem NLP Tracer Study IndoBERT.
Menyediakan endpoint inferensi tunggal, pemrosesan batch file (Excel/CSV),
Active Learning queue & human review, pemicu Continuous Retraining, dan analitik performa.
"""

import os
import io
import time
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from src.predict import TracerStudyPredictor
from src.model import TOPIC_ID2LABEL, SENTIMENT_ID2LABEL, PRETRAINED_MODEL_NAME
from src.preprocessing import clean_text

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Tracer Study NLP API - Dual-Model IndoBERT",
    description="Production REST API untuk Klasifikasi Topik Kritik/Saran & Analisis Sentimen Testimoni Alumni Universitas Dinamika.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Konfigurasi CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor: Optional[TracerStudyPredictor] = None
SERVER_START_TIME = datetime.datetime.now()

# Direktori Data & Review
ACTIVE_LEARNING_DIR = "data/active_learning"
ACTIVE_LEARNING_QUEUE_FILE = os.path.join(ACTIVE_LEARNING_DIR, "review_queue.json")
ACTIVE_LEARNING_REVIEWED_FILE = os.path.join(ACTIVE_LEARNING_DIR, "reviewed_samples.csv")
os.makedirs(ACTIVE_LEARNING_DIR, exist_ok=True)


# ==========================================
# PYDANTIC SCHEMAS (Request & Response)
# ==========================================

class SinglePredictRequest(BaseModel):
    text: str = Field(..., description="Teks masukan kritik, saran, atau testimoni alumni", min_length=1)
    threshold: Optional[float] = Field(0.70, description="Ambang batas confidence untuk active learning flag")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Fasilitas komputer di lab programming perlu diupgrade karena speknya sudah lambat untuk render.",
                "threshold": 0.70
            }
        }
    }


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="Daftar teks kuesioner untuk diproses sekaligus")
    threshold: Optional[float] = Field(0.70, description="Ambang batas confidence")

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Kurikulum sudah sangat relevan dengan kebutuhan industri software saat ini.",
                    "Mohon pelayanan biro administrasi kemahasiswaan bisa lebih cepat dan tanggap."
                ],
                "threshold": 0.70
            }
        }
    }


class ActiveLearningReviewRequest(BaseModel):
    queue_id: str = Field(..., description="ID unik antrean masukan")
    original_text: str = Field(..., description="Teks asli kuesioner")
    corrected_topic_id: int = Field(..., description="ID Topik hasil koreksi manusia (0-4)", ge=0, le=4)
    corrected_sentiment_id: int = Field(..., description="ID Sentimen hasil koreksi manusia (0-1)", ge=0, le=1)
    reviewer_notes: Optional[str] = Field(None, description="Catatan verifikator / reviewer")


# ==========================================
# LIFECYCLE & HELPER FUNCTIONS
# ==========================================

def get_or_create_predictor() -> TracerStudyPredictor:
    """Mengambil atau menginisialisasi instans model inferensi terpadu."""
    global predictor
    if predictor is None:
        topic_dir = "models/topic_model"
        sentiment_dir = "models/sentiment_model"
        predictor = TracerStudyPredictor(
            topic_model_dir=topic_dir,
            sentiment_model_dir=sentiment_dir,
        )
    return predictor


def load_review_queue() -> List[Dict[str, Any]]:
    """Memuat antrean masukan yang membutuhkan review manusia."""
    if not os.path.exists(ACTIVE_LEARNING_QUEUE_FILE):
        return []
    try:
        with open(ACTIVE_LEARNING_QUEUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_review_queue(queue: List[Dict[str, Any]]) -> None:
    """Menyimpan pembaruan antrean review."""
    with open(ACTIVE_LEARNING_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def add_to_review_queue(text: str, prediction_result: Dict[str, Any], reason: str = "Low Confidence (< 70%)") -> None:
    """Menambahkan teks hasil inferensi dengan confidence rendah ke antrean review."""
    queue = load_review_queue()
    
    # Cek apakah teks sudah ada di antrean
    for item in queue:
        if item.get("text") == text:
            return

    new_item = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "cleaned_text": prediction_result.get("teks_bersih", ""),
        "predicted_topic": prediction_result.get("topic"),
        "predicted_sentiment": prediction_result.get("sentiment"),
        "reason": reason,
        "status": "pending_review",
    }
    queue.insert(0, new_item)
    # Simpan maksimal 200 antrean terbaru
    save_review_queue(queue[:200])


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.on_event("startup")
def startup_event():
    """Memuat model IndoBERT saat server API pertama kali dihidupkan."""
    try:
        get_or_create_predictor()
        print("🚀 [FastAPI] IndoBERT Dual-Model siap melayani request inferensi.")
    except Exception as e:
        print(f"⚠️ [FastAPI] Peringatan saat startup: {e}")


@app.get("/health", summary="Health Check Sistem & Status Model")
def health_check():
    """
    Endpoint pemantau kondisi sistem, memori, status GPU/CPU, dan ketersediaan model IndoBERT.
    """
    pred = get_or_create_predictor()
    device_str = str(pred.device)
    has_cuda = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU Execution)"
    uptime = str(datetime.datetime.now() - SERVER_START_TIME).split(".")[0]

    return {
        "status": "online",
        "service_name": "Universitas Dinamika - Tracer Study NLP Inference Engine",
        "backbone_model": PRETRAINED_MODEL_NAME,
        "device": device_str,
        "cuda_available": has_cuda,
        "cuda_device_name": cuda_name,
        "topic_model_loaded": pred.topic_model is not None,
        "sentiment_model_loaded": pred.sentiment_model is not None,
        "server_uptime": uptime,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.post("/api/predict", summary="Inferensi Tunggal (Topik & Sentimen)")
def predict_single(request: SinglePredictRequest):
    """
    Menerima satu teks kuesioner alumni, membersihkan teks secara otomatis,
    dan mengembalikan klasifikasi topik (5 kelas), analisis sentimen (2 kelas),
    serta probabilitas per kelas dan flag active learning.
    """
    start_time = time.time()
    pred = get_or_create_predictor()
    pred.threshold = request.threshold

    result = pred.predict(request.text)
    latency_ms = round((time.time() - start_time) * 1000, 2)
    result["latency_ms"] = latency_ms

    # Jika butuh review, catat otomatis ke queue active learning
    if result.get("requires_human_review", False):
        add_to_review_queue(request.text, result)

    return result


@app.post("/api/predict/batch", summary="Inferensi Sekaligus (Batch List)")
def predict_batch(request: BatchPredictRequest):
    """
    Menerima daftar teks kuesioner alumni, menjalankan inferensi berkecepatan tinggi,
    dan menyajikan hasil prediksi beserta rekapitulasi distribusi agregat.
    """
    start_time = time.time()
    pred = get_or_create_predictor()
    pred.threshold = request.threshold

    results = []
    topic_counts = {label: 0 for label in TOPIC_ID2LABEL.values()}
    sentiment_counts = {label: 0 for label in SENTIMENT_ID2LABEL.values()}
    review_needed_count = 0

    for text in request.texts:
        res = pred.predict(text)
        results.append(res)
        
        if res.get("topic") and res["topic"].get("label_name") in topic_counts:
            topic_counts[res["topic"]["label_name"]] += 1
            
        if res.get("sentiment") and res["sentiment"].get("label_name") in sentiment_counts:
            sentiment_counts[res["sentiment"]["label_name"]] += 1
            
        if res.get("requires_human_review", False):
            review_needed_count += 1
            add_to_review_queue(text, res)

    total_latency_ms = round((time.time() - start_time) * 1000, 2)
    avg_latency_ms = round(total_latency_ms / max(len(request.texts), 1), 2)

    return {
        "total_items": len(request.texts),
        "total_latency_ms": total_latency_ms,
        "average_latency_per_item_ms": avg_latency_ms,
        "topic_distribution": topic_counts,
        "sentiment_distribution": sentiment_counts,
        "items_requiring_human_review": review_needed_count,
        "predictions": results,
    }


@app.post("/api/upload", summary="Upload & Anotasi File Tracer Study (Excel / CSV)")
async def upload_and_process_file(
    file: UploadFile = File(..., description="File Excel (.xlsx/.xls) atau CSV (.csv) kuesioner"),
    text_column: Optional[str] = Form(None, description="Nama kolom teks (opsional, auto-detect jika kosong)"),
    threshold: float = Form(0.70, description="Ambang batas keyakinan active learning"),
):
    """
    Mengunggah dataset kuesioner Tracer Study mentah dalam format Excel atau CSV,
    mendeteksi kolom masukan teks secara otomatis, melakukan inferensi dual-model batch,
    dan mengembalikan statistik hasil klasifikasi serta URL/ID untuk mengunduh file beranotasi.
    """
    filename = file.filename or "uploaded_tracer_study"
    contents = await file.read()
    
    # 1. Parsing File ke DataFrame
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Format file harus .xlsx, .xls, atau .csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File yang diunggah kosong")

    # 2. Deteksi Kolom Teks
    target_col = None
    if text_column and text_column in df.columns:
        target_col = text_column
    else:
        # Auto-detect kolom teks yang paling mungkin
        possible_names = ["teks", "text", "kritik", "saran", "testimoni", "masukan", "komentar", "feedback", "uraian"]
        for col in df.columns:
            if any(p in str(col).lower() for p in possible_names):
                target_col = col
                break
        if target_col is None:
            # Pilih kolom bertipe object/string pertama
            for col in df.columns:
                if df[col].dtype == "object":
                    target_col = col
                    break
    
    if target_col is None:
        raise HTTPException(status_code=400, detail="Tidak ditemukan kolom teks yang valid dalam file")

    # 3. Jalankan Prediksi Batch
    pred = get_or_create_predictor()
    pred.threshold = threshold

    predictions_data = []
    pred_topics = []
    pred_topic_confs = []
    pred_sentiments = []
    pred_sentiment_confs = []
    needs_review_list = []

    for idx, row in df.iterrows():
        raw_val = str(row[target_col]) if pd.notna(row[target_col]) else ""
        if not raw_val.strip() or raw_val.lower() == "nan":
            pred_topics.append("Tidak Ada Teks")
            pred_topic_confs.append(0.0)
            pred_sentiments.append("Tidak Ada Teks")
            pred_sentiment_confs.append(0.0)
            needs_review_list.append(False)
            continue

        res = pred.predict(raw_val)
        t_label = res["topic"]["label_name"] if res.get("topic") else "N/A"
        t_conf = res["topic"]["confidence"] if res.get("topic") else 0.0
        s_label = res["sentiment"]["label_name"] if res.get("sentiment") else "N/A"
        s_conf = res["sentiment"]["confidence"] if res.get("sentiment") else 0.0
        rev = res.get("requires_human_review", False)

        pred_topics.append(t_label)
        pred_topic_confs.append(t_conf)
        pred_sentiments.append(s_label)
        pred_sentiment_confs.append(s_conf)
        needs_review_list.append(rev)

        if rev:
            add_to_review_queue(raw_val, res)

        predictions_data.append({
            "row_index": idx,
            "text": raw_val,
            "topic": t_label,
            "topic_confidence": t_conf,
            "sentiment": s_label,
            "sentiment_confidence": s_conf,
            "requires_review": rev,
        })

    # Tambahkan kolom hasil ke DataFrame
    annotated_df = df.copy()
    annotated_df["Predicted_Topic"] = pred_topics
    annotated_df["Topic_Confidence"] = pred_topic_confs
    annotated_df["Predicted_Sentiment"] = pred_sentiments
    annotated_df["Sentiment_Confidence"] = pred_sentiment_confs
    annotated_df["Flag_Review_Manual"] = ["YA" if r else "TIDAK" for r in needs_review_list]

    # Simpan file hasil ke temporary in-memory buffer
    out_dir = "reports/annotated_uploads"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"annotated_{uuid.uuid4().hex[:8]}_{filename}"
    if not out_filename.endswith((".xlsx", ".csv")):
        out_filename += ".xlsx"
    
    out_path = os.path.join(out_dir, out_filename)
    if out_filename.endswith(".csv"):
        annotated_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        annotated_df.to_excel(out_path, index=False, engine="openpyxl")

    # Hitung ringkasan statistik
    topic_summary = annotated_df["Predicted_Topic"].value_counts().to_dict()
    sentiment_summary = annotated_df["Predicted_Sentiment"].value_counts().to_dict()

    return {
        "status": "success",
        "processed_filename": filename,
        "detected_text_column": target_col,
        "total_rows_processed": len(df),
        "total_requires_review": sum(needs_review_list),
        "topic_summary": topic_summary,
        "sentiment_summary": sentiment_summary,
        "sample_preview": predictions_data[:15],
        "download_url": f"/api/download/annotated/{out_filename}",
    }


@app.get("/api/download/annotated/{file_name}", summary="Unduh File Beranotasi Hasil Batch")
def download_annotated_file(file_name: str):
    """Mengunduh file Excel atau CSV yang telah selesai diolah dan diberi label oleh model IndoBERT."""
    out_dir = "reports/annotated_uploads"
    file_path = os.path.join(out_dir, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File hasil anotasi tidak ditemukan atau telah kedaluwarsa")

    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if file_name.endswith(".xlsx") else "text/csv"
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type=media_type,
    )


# ==========================================
# ACTIVE LEARNING & HUMAN-IN-THE-LOOP ENDPOINTS
# ==========================================

@app.get("/api/active-learning/queue", summary="Daftar Antrean Masukan Butuh Review")
def get_active_learning_queue():
    """
    Mengambil daftar teks kritik, saran, atau testimoni yang memiliki skor keyakinan rendah (< 70%)
    untuk ditinjau dan dikoreksi oleh administrator / staf Tracer Study.
    """
    queue = load_review_queue()
    return {
        "total_in_queue": len(queue),
        "queue": queue,
    }


@app.post("/api/active-learning/review", summary="Simpan Koreksi Label Manusia (Human-in-the-Loop)")
def submit_human_review(review: ActiveLearningReviewRequest):
    """
    Menyimpan hasil review dan koreksi label topik & sentimen oleh manusia.
    Data ini akan disimpan ke buffer dataset latihan (`data/active_learning/reviewed_samples.csv`)
    sebagai amunisi untuk siklus Continuous Retraining berikutnya.
    """
    queue = load_review_queue()
    updated_queue = [item for item in queue if item.get("id") != review.queue_id]
    save_review_queue(updated_queue)

    # Simpan ke CSV buffer
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "teks_asli": review.original_text,
        "teks_bersih": clean_text(review.original_text),
        "corrected_topic_id": review.corrected_topic_id,
        "corrected_topic_name": TOPIC_ID2LABEL.get(review.corrected_topic_id, "Unknown"),
        "corrected_sentiment_id": review.corrected_sentiment_id,
        "corrected_sentiment_name": SENTIMENT_ID2LABEL.get(review.corrected_sentiment_id, "Unknown"),
        "reviewer_notes": review.reviewer_notes or "",
    }

    df_new = pd.DataFrame([row])
    if os.path.exists(ACTIVE_LEARNING_REVIEWED_FILE):
        df_new.to_csv(ACTIVE_LEARNING_REVIEWED_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(ACTIVE_LEARNING_REVIEWED_FILE, mode="w", header=True, index=False, encoding="utf-8-sig")

    # Hitung total yang sudah terkumpul
    try:
        df_reviewed = pd.read_csv(ACTIVE_LEARNING_REVIEWED_FILE)
        total_reviewed = len(df_reviewed)
    except Exception:
        total_reviewed = 1

    return {
        "status": "success",
        "message": f"Koreksi berhasil disimpan ke buffer Active Learning (Total: {total_reviewed} sampel terverifikasi).",
        "remaining_in_queue": len(updated_queue),
        "sample": row,
    }


@app.post("/api/retrain/trigger", summary="Memicu Siklus Continuous Retraining")
def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Memicu proses continuous retraining model IndoBERT menggunakan akumulasi data verifikasi baru.
    Menerapkan perlindungan anti-regresi: model baru hanya menggantikan model lama jika performa F1 >= model lama.
    """
    from src.continuous_retraining import run_continuous_retraining_pipeline
    
    # Jalankan sebagai asynchronous background task agar API tidak freeze
    background_tasks.add_task(run_continuous_retraining_pipeline)
    
    return {
        "status": "triggered",
        "message": "Pipeline Continuous Retraining sedang dieksekusi di background. Log dan status checkpoint akan diperbarui secara otomatis.",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/api/analytics/summary", summary="Ringkasan Analitik, Metrik & Visualisasi")
def get_analytics_summary():
    """
    Menyajikan metrik evaluasi model, daftar gambar WordCloud per topik,
    dan statistik sebaran masukan untuk dasbor visual pimpinan.
    """
    metrics_path = "reports/metrics_summary.json"
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    # Cek WordCloud files
    wc_dir = "reports/wordclouds"
    available_wordclouds = []
    if os.path.exists(wc_dir):
        for root, _, files in os.walk(wc_dir):
            for fname in sorted(files):
                if fname.endswith(".png"):
                    rel_path = os.path.relpath(os.path.join(root, fname), "reports")
                    clean_title = fname.replace("wordcloud_", "").replace(".png", "").replace("___", " & ").replace("__", " - ").replace("_", " ").title()
                    available_wordclouds.append({
                        "title": clean_title,
                        "file_name": fname,
                        "url": f"/reports/{rel_path}",
                    })

    # Cek Figure files (Confusion Matrix, dsb)
    fig_dir = "reports/figures"
    available_figures = []
    if os.path.exists(fig_dir):
        for fname in sorted(os.listdir(fig_dir)):
            if fname.endswith(".png"):
                available_figures.append({
                    "title": fname.replace(".png", "").replace("_", " ").title(),
                    "file_name": fname,
                    "url": f"/reports/figures/{fname}",
                })

    # Review queue length
    q_len = len(load_review_queue())
    reviewed_count = 0
    if os.path.exists(ACTIVE_LEARNING_REVIEWED_FILE):
        try:
            reviewed_count = len(pd.read_csv(ACTIVE_LEARNING_REVIEWED_FILE))
        except Exception:
            reviewed_count = 0

    return {
        "metrics": metrics_data,
        "wordclouds": available_wordclouds,
        "figures": available_figures,
        "active_learning_stats": {
            "pending_in_queue": q_len,
            "total_human_verified_samples": reviewed_count,
            "min_samples_for_retraining": 10,
        },
        "topic_labels": TOPIC_ID2LABEL,
        "sentiment_labels": SENTIMENT_ID2LABEL,
    }


# ==========================================
# STATIC FILES & WEB DASHBOARD MOUNTING
# ==========================================

# Mount direktori reports untuk visualisasi gambar
if os.path.exists("reports"):
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Mount direktori static untuk Web UI Dashboard
STATIC_DIR = "src/static"
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", summary="Web Dashboard UI")
    def serve_dashboard():
        """Menyajikan halaman Web Dashboard Interaktif."""
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Web Dashboard HTML belum diinisialisasi."}
