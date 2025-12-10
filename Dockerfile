FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/ /app/scripts/

# Create directories
RUN mkdir -p /app/output /app/logs /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]