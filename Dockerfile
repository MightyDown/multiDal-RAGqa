# Stage 1: Build Vue frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API
FROM python:3.11-slim

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.ustc.edu.cn/pypi/web/simple -r requirements.txt

COPY . .
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.multidal.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
