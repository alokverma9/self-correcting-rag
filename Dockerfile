FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg, pgvector, and native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and assets
COPY . .

# Expose API port
EXPOSE 8000

ENV PORT=8000

# Start production uvicorn server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
