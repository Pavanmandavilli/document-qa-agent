FROM python:3.11-slim

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app

RUN useradd --create-home appuser \
    && mkdir -p /app/logs /docs \
    && chown -R appuser /opt/hf /app/logs /docs
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
