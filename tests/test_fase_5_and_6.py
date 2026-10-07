"""
Skrip Uji Otomatis untuk Verifikasi Fase 5 & 6 (CRISP-DM 6):
Menguji REST API Serving, Endpoints, Batch Processing, Active Learning, dan UI Static Files.
"""

import os
import sys
import json
import io
import pandas as pd
from fastapi.testclient import TestClient

# Pastikan root workspace terdaftar di sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api import app

client = TestClient(app)

def test_health_endpoint():
    print("\n🔍 [1/6] Menguji GET /health ...")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "online"
    assert "backbone_model" in data
    print("✅ GET /health berhasil:", data["status"], "| Model loaded:", data["topic_model_loaded"])

def test_single_predict_endpoint():
    print("\n🔍 [2/6] Menguji POST /api/predict ...")
    payload = {
        "text": "Fasilitas lab komputer mohon diupgrade RAM dan GPU untuk render 3D.",
        "threshold": 0.70
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "topic" in data
    assert "sentiment" in data
    assert "latency_ms" in data
    print("✅ POST /api/predict berhasil:")
    print(f"   - Teks Asli: {data['teks_asli']}")
    print(f"   - Prediksi Topik: {data['topic']['label_name']} ({data['topic']['confidence_percentage']})")
    print(f"   - Prediksi Sentimen: {data['sentiment']['label_name']} ({data['sentiment']['confidence_percentage']})")
    print(f"   - Latency: {data['latency_ms']} ms")

def test_batch_predict_endpoint():
    print("\n🔍 [3/6] Menguji POST /api/predict/batch ...")
    payload = {
        "texts": [
            "Kurikulum materi praktikum AI dan cloud computing sangat relevan dengan dunia kerja.",
            "Dosen pembimbing sangat ramah dan suportif saat bimbingan skripsi.",
            "Pelayanan biro administrasi kampus mohon dipercepat."
        ],
        "threshold": 0.70
    }
    response = client.post("/api/predict/batch", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["total_items"] == 3
    assert "topic_distribution" in data
    assert "sentiment_distribution" in data
    print("✅ POST /api/predict/batch berhasil:", f"Total: {data['total_items']} teks | Waktu: {data['total_latency_ms']} ms")

def test_file_upload_endpoint():
    print("\n🔍 [4/6] Menguji POST /api/upload (Excel & CSV) ...")
    # Buat dummy DataFrame
    df = pd.DataFrame({
        "No": [1, 2, 3],
        "Uraian_Kritik_Saran": [
            "Fasilitas wifi kampus sering lambat di perpustakaan.",
            "Perbanyak info lowongan kerja dan magang dari mitra industri.",
            "Materi kuliah kurikulum sangat bagus dan terkini."
        ]
    })
    
    # Konversi ke bytes Excel
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)

    files = {"file": ("test_kuesioner.xlsx", excel_buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"threshold": "0.70"}

    response = client.post("/api/upload", files=files, data=data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"
    res_data = response.json()
    assert res_data["total_rows_processed"] == 3
    assert "download_url" in res_data
    print("✅ POST /api/upload berhasil:", f"Total baris: {res_data['total_rows_processed']} | Download URL: {res_data['download_url']}")

def test_active_learning_endpoints():
    print("\n🔍 [5/6] Menguji Active Learning Queue & Review Endpoint ...")
    # 1. Get queue
    q_resp = client.get("/api/active-learning/queue")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    print("✅ GET /api/active-learning/queue berhasil:", f"Total antrean: {q_data['total_in_queue']}")

    # 2. Submit human review
    review_payload = {
        "queue_id": "test_id_001",
        "original_text": "Wifi perpustakaan sering putus saat malam hari.",
        "corrected_topic_id": 1,  # Fasilitas
        "corrected_sentiment_id": 0,  # Negatif
        "reviewer_notes": "Verifikasi unit test"
    }
    r_resp = client.post("/api/active-learning/review", json=review_payload)
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["status"] == "success"
    print("✅ POST /api/active-learning/review berhasil:", r_data["message"])

def test_analytics_and_static_ui():
    print("\n🔍 [6/6] Menguji Analytics Summary & Web Dashboard UI ...")
    # 1. Analytics
    a_resp = client.get("/api/analytics/summary")
    assert a_resp.status_code == 200
    a_data = a_resp.json()
    assert "metrics" in a_data
    assert "wordclouds" in a_data
    print("✅ GET /api/analytics/summary berhasil: Wordclouds count =", len(a_data["wordclouds"]))

    # 2. UI Dashboard
    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "<!DOCTYPE html>" in ui_resp.text
    print("✅ GET / (Web Dashboard) berhasil:", f"HTML Size: {len(ui_resp.text)} bytes")

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 MEMULAI RANGKAIAN UJI INTEGRASI FASE 5 & 6 (CRISP-DM 6)")
    print("=" * 70)
    
    test_health_endpoint()
    test_single_predict_endpoint()
    test_batch_predict_endpoint()
    test_file_upload_endpoint()
    test_active_learning_endpoints()
    test_analytics_and_static_ui()

    print("\n" + "=" * 70)
    print("🎉 SEMUA TEST FASE 5 & 6 BERHASIL DILALUI DENGAN SUKSES! (6/6)")
    print("=" * 70)
