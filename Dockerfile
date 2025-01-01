FROM python:3.11.4-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install --no-install-recommends -y tzdata && \
    pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /app/logs

COPY . .

WORKDIR /app/src

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "360" ]
