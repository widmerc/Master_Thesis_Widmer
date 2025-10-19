FROM python:3.11-slim

LABEL maintainer="widmerc"

# Install system deps required by geopandas / gdal / spatial index
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gdal-bin \
        libgdal-dev \
        libspatialindex-dev \
        libgeos-dev \
        git \
        curl \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install deps
RUN python -m pip install --upgrade pip setuptools wheel; \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

# Expose port and use a reasonably small default worker setup
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]