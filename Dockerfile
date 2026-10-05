FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY documents ./documents
COPY scripts ./scripts

RUN PYTHONPATH=/app python scripts/ingest_documents.py
CMD ["python", "-m", "app.slack.bot"]