"""
Server runner & entrypoint untuk sistem REST API FastAPI IndoBERT Tracer Study.
Mendukung pengaturan host, port, reload mode, dan worker configuration.
"""

import os
import sys
import argparse
import uvicorn

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False, workers: int = 1):
    """
    Menjalankan Uvicorn ASGI Web Server untuk FastAPI application.
    """
    print("=" * 70)
    print("🚀 UNIVERSITAS DINAMIKA - NLP TRACER STUDY SERVING SYSTEM")
    print(f"📡 API Server: http://{host}:{port}")
    print(f"📚 Swagger UI Docs: http://localhost:{port}/docs")
    print(f"📊 Web Dashboard UI: http://localhost:{port}/")
    print("=" * 70)

    # Pastikan current working directory ada di sys.path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan FastAPI Serving Server untuk IndoBERT Tracer Study.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Aktifkan auto-reload saat pengembangan")
    parser.add_argument("--workers", type=int, default=1, help="Jumlah ASGI workers (default: 1)")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload, workers=args.workers)
