# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8771 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MODEL_PATH=model/transfer_model.keras

WORKDIR /srv

# libgomp1 : requis par TensorFlow ; curl : sonde de sante HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + frontend + modele (48 Mo) + images du jeu.
COPY app ./app
COPY web ./web
COPY model ./model
COPY data ./data

EXPOSE 8771

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
