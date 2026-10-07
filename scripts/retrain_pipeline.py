"""
Skrip CLI untuk menjalankan pipeline Continuous Retraining & Active Learning Feedback Loop.
Dapat dijalankan secara mandiri atau dijadwalkan via cron job kampus / task scheduler.
"""

import sys
import os

# Pastikan workspace root ada di path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.continuous_retraining import run_continuous_retraining_pipeline

def main():
    print("🚀 [CLI] Memulai Continuous Retraining Pipeline...")
    res = run_continuous_retraining_pipeline()
    print("\n📊 Hasil Eksekusi Pipeline:")
    print(f"- Timestamp: {res.get('timestamp')}")
    print(f"- Waktu Komputasi: {res.get('elapsed_seconds')} detik")
    print(f"- Status Model Topik: {res.get('topic_comparison', {}).get('decision')}")
    print(f"- Status Model Sentimen: {res.get('sentiment_comparison', {}).get('decision')}")

if __name__ == "__main__":
    main()
