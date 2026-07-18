FROM python:3.10-slim

WORKDIR /app

# Install compilation dependency binaries inside Linux base image
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose network boundaries
EXPOSE 8000
EXPOSE 8501

# Script to fire up both microservices simultaneously upon loading initialization
CMD uvicorn src.main:app --host 0.0.0.0 --port 8000 & streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
