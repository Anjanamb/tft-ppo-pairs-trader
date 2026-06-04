# TFT + PPO Pairs Trading System — dashboard image
FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# CPU-only PyTorch first so the rest of requirements resolve against it and the
# image stays small (no CUDA runtime).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Writable mounts for the database, logs and trained artifacts.
RUN mkdir -p data logs models

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
