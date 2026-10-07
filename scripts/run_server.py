"""
Skrip peluncur cepat Web Server REST API FastAPI & Web Dashboard UI Tracer Study IndoBERT.
"""

import os
import sys

# Tambahkan root path proyek
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.server import run_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Jalankan Server FastAPI Tracer Study.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto reload")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
