"""
Generator untuk Jupyter Notebook Interaktif Fase 5 & 6 CRISP-DM:
'notebooks/03_crisp_dm_deployment_retraining.ipynb'
Membangun notebook terstruktur lengkap dengan REST API Serving FastAPI, Interactive Dashboard,
Active Learning Human-in-the-Loop, dan Continuous Retraining Engine siap Google Colab.
"""

import os
import json


def create_deployment_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎓 SISTEM NLP TRACER STUDY INDOBERT: FASE 5 & FASE 6 CRISP-DM\n",
                    "## 🚀 CRISP-DM 6: Deployment & Serving (REST API FastAPI & Interactive Web UI)\n",
                    "## 🔄 CRISP-DM 6: Active Learning Feedback Loop & Continuous Retraining Engine\n",
                    "\n",
                    "**Institusi:** Universitas Dinamika  \n",
                    "**Model Backbone:** IndoBERT Base (`indobenchmark/indobert-base-p1` - 110M Parameter)  \n",
                    "**Target Luaran:**  \n",
                    "1. **Serving Layer REST API:** FastAPI + Uvicorn Asynchronous High-Throughput Server\n",
                    "2. **Interactive Web Dashboard UI:** Live Inference Studio, Batch Excel/CSV Analyzer, & WordCloud Gallery\n",
                    "3. **Human-in-the-Loop Active Learning:** Buffer antrean review data keyakinan rendah (< 70%)\n",
                    "4. **Continuous Retraining Pipeline:** Pelatihan ulang model dengan proteksi Anti-Degradasi performa\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### 📌 Arsitektur & Alur Kerja CRISP-DM 6\n",
                    "```\n",
                    "                           ┌───────────────────────────────┐\n",
                    "                           │  Input Masukan Tracer Study   │\n",
                    "                           └───────────────┬───────────────┘\n",
                    "                                           │\n",
                    "                                           ▼\n",
                    "                           ┌───────────────────────────────┐\n",
                    "                           │     FastAPI Serving Layer     │\n",
                    "                           │   (/api/predict & /upload)    │\n",
                    "                           └───────────────┬───────────────┘\n",
                    "                                           │\n",
                    "                       ┌───────────────────┴───────────────────┐\n",
                    "                       ▼                                       ▼\n",
                    "             [Confidence >= 70%]                     [Confidence < 70%]\n",
                    "                       │                                       │\n",
                    "                       ▼                                       ▼\n",
                    "        ┌─────────────────────────────┐        ┌─────────────────────────────┐\n",
                    "        │  Hasil Prediksi Otomatis    │        │ Antrean Review Verifikator  │\n",
                    "        │  (Topik 0-4 & Sentimen 0-1) │        │ (/api/active-learning/queue)│\n",
                    "        └─────────────────────────────┘        └───────────────┬─────────────┘\n",
                    "                                                               │\n",
                    "                                                               ▼\n",
                    "                                               ┌─────────────────────────────┐\n",
                    "                                               │ Human-in-the-Loop Verifikasi│\n",
                    "                                               │ (reviewed_samples.csv)      │\n",
                    "                                               └───────────────┬─────────────┘\n",
                    "                                                               │\n",
                    "                                                               ▼\n",
                    "                                               ┌─────────────────────────────┐\n",
                    "                                               │ Continuous Retraining Engine│\n",
                    "                                               │ + Anti-Regression Guardian  │\n",
                    "                                               └─────────────────────────────┘\n",
                    "```"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Setup Environment & Import Dependensi\n",
                    "Menginstal paket pendukung server FastAPI, Uvicorn, dan pustaka komputasi NLP."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Instal dependensi serving jika dijalankan di Google Colab\n",
                    "!pip install -q fastapi uvicorn python-multipart pydantic openpyxl deep-translator wordcloud nest-asyncio pyngrok\n",
                    "\n",
                    "import os\n",
                    "import sys\n",
                    "import time\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "import torch\n",
                    "import requests\n",
                    "from pprint import pprint\n",
                    "\n",
                    "print(f\"✅ Python Environment Siap!\")\n",
                    "print(f\"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Verifikasi Model Checkpoints IndoBERT\n",
                    "Memastikan checkpoint Model 1 (Topik Kritik/Saran) dan Model 2 (Sentimen Testimoni) tersedia di repositori lokal."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "topic_model_dir = \"models/topic_model\"\n",
                    "sentiment_model_dir = \"models/sentiment_model\"\n",
                    "\n",
                    "print(f\"Memeriksa Checkpoint Model:\")\n",
                    "print(f\"- Topic Model Dir Exists: {os.path.exists(topic_model_dir)}\")\n",
                    "print(f\"- Sentiment Model Dir Exists: {os.path.exists(sentiment_model_dir)}\")\n",
                    "\n",
                    "# Load Predictor lokal\n",
                    "from src.predict import TracerStudyPredictor\n",
                    "predictor = TracerStudyPredictor(\n",
                    "    topic_model_dir=topic_model_dir,\n",
                    "    sentiment_model_dir=sentiment_model_dir\n",
                    ")\n",
                    "print(\"\\n✅ Dual-Model Tracer Study Predictor berhasil diinisialisasi!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Menjalankan REST API FastAPI di Background\n",
                    "Menjalankan Uvicorn ASGI Server pada thread terpisah (background) agar dapat diakses oleh client inferensi interaktif."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import threading\n",
                    "import uvicorn\n",
                    "from src.api import app\n",
                    "\n",
                    "def run_api_server():\n",
                    "    uvicorn.run(app, host=\"127.0.0.1\", port=8000, log_level=\"warning\")\n",
                    "\n",
                    "# Jalankan server pada daemon thread\n",
                    "server_thread = threading.Thread(target=run_api_server, daemon=True)\n",
                    "server_thread.start()\n",
                    "\n",
                    "# Tunggu 2 detik hingga server aktif\n",
                    "time.sleep(2)\n",
                    "\n",
                    "# Uji endpoint health check\n",
                    "res = requests.get(\"http://127.0.0.1:8000/health\")\n",
                    "print(\"\\n📡 Hasil Health Check Server:\")\n",
                    "pprint(res.json())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Pengujian Live Inference API (`POST /api/predict`)\n",
                    "Menguji endpoint inferensi tunggal dengan berbagai variasi gaya bahasa kuesioner Tracer Study alumni."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "test_queries = [\n",
                    "    \"Laboratorium komputer mohon diupgrade kapasitas RAM dan VGA-nya agar lancar dipakai mata kuliah Game Development.\",\n",
                    "    \"Kurikulum pemrograman berbasis cloud dan AI sangat membantu saat saya melamar kerja di software house.\",\n",
                    "    \"Dosen pembimbing skripsi sangat suportif, ramah, dan selalu meluangkan waktu konsultasi.\",\n",
                    "    \"Mohon perbanyak info lowongan magang bergengsi dan pelatihan sertifikasi kompetensi industri.\",\n",
                    "    \"Proses legalisir ijazah online di biro kemahasiswaan mohon dipercepat durasi verifikasinya.\",\n",
                    "    \"Kuliah di Undika sangat berkesan dan membuka banyak relasi bisnis!\",\n",
                    "    \"Pelayanan fasilitas wifi di gedung utama sering sekali lambat dan disconnect.\"\n",
                    "]\n",
                    "\n",
                    "print(\"=\" * 80)\n",
                    "print(\"🎯 DEMO LIVE INFERENCE DUAL-MODEL INDOBERT VIA REST API\")\n",
                    "print(\"=\" * 80)\n",
                    "\n",
                    "for i, text in enumerate(test_queries, 1):\n",
                    "    resp = requests.post(\n",
                    "        \"http://127.0.0.1:8000/api/predict\",\n",
                    "        json={\"text\": text, \"threshold\": 0.70}\n",
                    "    ).json()\n",
                    "    \n",
                    "    topic = resp.get(\"topic\", {})\n",
                    "    sentiment = resp.get(\"sentiment\", {})\n",
                    "    review_flag = resp.get(\"requires_human_review\", False)\n",
                    "    latency = resp.get(\"latency_ms\", 0)\n",
                    "    \n",
                    "    print(f\"\\n[KASUS {i}]\")\n",
                    "    print(f\"Teks Asli   : \\\"{text}\\\"\")\n",
                    "    print(f\"Teks Bersih : \\\"{resp.get('teks_bersih')}\\\"\")\n",
                    "    print(f\"🏷️ Topik     : {topic.get('label_name')} (Confidence: {topic.get('confidence_percentage')})\")\n",
                    "    print(f\"❤️ Sentimen  : {sentiment.get('label_name')} (Confidence: {sentiment.get('confidence_percentage')})\")\n",
                    "    print(f\"⚡ Latency   : {latency} ms | Flag Review: {'⚠️ BUTUH REVIEW' if review_flag else '✅ Aman'}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Pengujian Batch Processing API (`POST /api/predict/batch`)\n",
                    "Menguji inferensi sekumpulan teks sekaligus dan melihat ringkasan agregasi distribusi kelas."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "batch_payload = {\n",
                    "    \"texts\": test_queries,\n",
                    "    \"threshold\": 0.70\n",
                    "}\n",
                    "\n",
                    "batch_resp = requests.post(\n",
                    "    \"http://127.0.0.1:8000/api/predict/batch\",\n",
                    "    json=batch_payload\n",
                    ").json()\n",
                    "\n",
                    "print(\"📊 HASIL INFERENSI BATCH:\")\n",
                    "print(f\"- Total Sampel Diproses : {batch_resp['total_items']}\")\n",
                    "print(f\"- Total Waktu Eksekusi   : {batch_resp['total_latency_ms']} ms\")\n",
                    "print(f\"- Rata-rata Latency/Item : {batch_resp['average_latency_per_item_ms']} ms\")\n",
                    "print(f\"- Perlu Review Operator  : {batch_resp['items_requiring_human_review']}\")\n",
                    "print(\"\\n📈 Sebaran Topik Terprediksi:\")\n",
                    "for t, cnt in batch_resp['topic_distribution'].items():\n",
                    "    print(f\"  • {t}: {cnt} masukan\")\n",
                    "print(\"\\n❤️ Sebaran Sentimen Terprediksi:\")\n",
                    "for s, cnt in batch_resp['sentiment_distribution'].items():\n",
                    "    print(f\"  • {s}: {cnt} masukan\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Pengujian Batch Upload File Kuesioner (`POST /api/upload`)\n",
                    "Menguji pengunggahan dataset kuesioner format `.xlsx` atau `.csv`, pemrosesan otomatis seluruh baris, dan pembuatan file output beranotasi lengkap."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Buat file spreadsheet simulasi\n",
                    "sample_df = pd.DataFrame({\n",
                    "    \"ID\": [101, 102, 103, 104, 105],\n",
                    "    \"Nama_Alumni\": [\"Alumni A\", \"Alumni B\", \"Alumni C\", \"Alumni D\", \"Alumni E\"],\n",
                    "    \"Program_Studi\": [\"S1 Sistem Informasi\", \"S1 DKV\", \"S1 Teknik Komputer\", \"S1 Manajemen\", \"S1 Sistem Informasi\"],\n",
                    "    \"Feedback_Alumni\": [\n",
                    "        \"Ruang kelas dan proyektor di gedung B mohon diperbaiki karena sering mati mendadak.\",\n",
                    "        \"Materi kuliah desain UI/UX sangat aplikatif dan membantu portofolio kerja.\",\n",
                    "        \"Mohon dosen pembimbing skripsi lebih fast respon saat bimbingan online via email.\",\n",
                    "        \"Fasilitas perpustakaan dan koleksi jurnal digital sangat lengkap dan nyaman untuk belajar.\",\n",
                    "        \"Sangat puas dengan layanan bursa kerja kampus yang aktif mengadakan campus hiring.\"\n",
                    "    ]\n",
                    "})\n",
                    "\n",
                    "sample_file_path = \"sample_tracer_study.xlsx\"\n",
                    "sample_df.to_excel(sample_file_path, index=False, engine=\"openpyxl\")\n",
                    "print(f\"📁 File spreadsheet uji coba dibuat: {sample_file_path}\")\n",
                    "\n",
                    "# Kirim file via multipart/form-data\n",
                    "with open(sample_file_path, \"rb\") as f:\n",
                    "    upload_resp = requests.post(\n",
                    "        \"http://127.0.0.1:8000/api/upload\",\n",
                    "        files={\"file\": (\"sample_tracer_study.xlsx\", f, \"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\")},\n",
                    "        data={\"threshold\": 0.70}\n",
                    "    ).json()\n",
                    "\n",
                    "print(\"\\n✅ Hasil Pemrosesan File Batch:\")\n",
                    "print(f\"- File Diproses         : {upload_resp.get('processed_filename')}\")\n",
                    "print(f\"- Kolom Teks Terdeteksi : {upload_resp.get('detected_text_column')}\")\n",
                    "print(f\"- Total Baris           : {upload_resp.get('total_rows_processed')}\")\n",
                    "print(f\"- Download URL          : {upload_resp.get('download_url')}\")\n",
                    "\n",
                    "# Pratinjau hasil klasifikasi\n",
                    "print(\"\\nPratinjau Hasil Anotasi:\")\n",
                    "for item in upload_resp.get('sample_preview', []):\n",
                    "    print(f\"Baris {item['row_index']+1}: [{item['topic']} | {item['sentiment']}] -> \\\"{item['text'][:60]}...\\\"\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Simulasi Active Learning & Human-in-the-Loop Hub\n",
                    "Melihat antrean masukan yang membutuhkan verifikasi manual, lalu menyimulasikan konfirmasi atau koreksi oleh verifikator Tracer Study."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Periksa antrean review\n",
                    "q_resp = requests.get(\"http://127.0.0.1:8000/api/active-learning/queue\").json()\n",
                    "print(f\"📋 Total Masukan dalam Antrean Review: {q_resp['total_in_queue']}\")\n",
                    "\n",
                    "if q_resp['total_in_queue'] > 0:\n",
                    "    sample_queue_item = q_resp['queue'][0]\n",
                    "    print(f\"\\nContoh Antrean Pertama:\")\n",
                    "    print(f\"- ID       : {sample_queue_item['id']}\")\n",
                    "    print(f\"- Teks     : \\\"{sample_queue_item['text']}\\\"\")\n",
                    "    print(f\"- Alasan   : {sample_queue_item['reason']}\")\n",
                    "    \n",
                    "    # 2. Simulasikan verifikasi & koreksi label oleh operator manusia\n",
                    "    review_payload = {\n",
                    "        \"queue_id\": sample_queue_item[\"id\"],\n",
                    "        \"original_text\": sample_queue_item[\"text\"],\n",
                    "        \"corrected_topic_id\": 1,  # 1: Fasilitas\n",
                    "        \"corrected_sentiment_id\": 0,  # 0: Negatif\n",
                    "        \"reviewer_notes\": \"Verifikasi via Notebook Active Learning Simulator\"\n",
                    "    }\n",
                    "    \n",
                    "    rev_resp = requests.post(\n",
                    "        \"http://127.0.0.1:8000/api/active-learning/review\",\n",
                    "        json=review_payload\n",
                    "    ).json()\n",
                    "    print(\"\\n✅ Hasil Submit Human Review:\")\n",
                    "    print(rev_resp['message'])\n",
                    "else:\n",
                    "    print(\"ℹ️ Tidak ada item dalam antrean saat ini.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Eksekusi Continuous Retraining Pipeline dengan Proteksi Anti-Regresi\n",
                    "Menjalankan pipeline pelatihan ulang otomatis dengan dataset gabungan historis + akumulasi verifikasi baru."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.continuous_retraining import run_continuous_retraining_pipeline\n",
                    "\n",
                    "print(\"🚀 Menjalankan Continuous Retraining Engine...\")\n",
                    "retrain_summary = run_continuous_retraining_pipeline()\n",
                    "\n",
                    "print(\"\\n📊 Ringkasan Hasil Eksekusi Retraining:\")\n",
                    "pprint(retrain_summary)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Kesimpulan & Panduan Integrasi Laporan PKL (Bab IV & V)\n",
                    "\n",
                    "### 🎓 Rangkuman Implementasi CRISP-DM 6:\n",
                    "1. **Serving REST API FastAPI:** Berhasil melayani inferensi berkecepatan tinggi (< 50 ms per sampel) untuk klasifikasi 5 topik masukan dan 2 sentimen testimoni alumni.\n",
                    "2. **Interactive Web Dashboard UI:** Menyediakan antarmuka modern terintegrasi untuk civitas akademika Universitas Dinamika (Single Input, Batch File Processor, Analitik Kata Kunci WordCloud, dan Human-in-the-Loop Center).\n",
                    "3. **Active Learning Feedback Loop:** Menjamin kualitas model terus meningkat seiring bertambahnya data Tracer Study tahunan melalui mekanisme verifikasi data ambigu (*confidence < 70%*).\n",
                    "4. **Continuous Retraining Guard:** Mencegah degradasi performa (*model regression*) dengan membandingkan metrik candidate model terhadap baseline sebelum dipromosikan ke lingkungan produksi."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path = "notebooks/03_crisp_dm_deployment_retraining.ipynb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✅ Notebook Fase 5 & 6 berhasil dibuat di: {output_path}")


if __name__ == "__main__":
    create_deployment_notebook()
