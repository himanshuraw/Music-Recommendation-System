## Updated ml_service Dockerfile with volume declarations for proper Compose merging

# Builder stage
FROM python:3.9-slim as builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY main.py .
COPY entrypoint.sh .

# Copy initial data and models (these will be used to initialize volumes)
COPY data/ /app/build_data/
COPY models/ /app/build_models/

# Declare mount points so that Compose sees the volumes in image metadata
VOLUME ["/app/data", "/app/models"]

# Create directories that will be volume-mounted (fallback)
RUN mkdir -p /app/data /app/models

# Set permissions and switch to non-root user
RUN chmod +x /app/entrypoint.sh && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

# Use entrypoint script that will handle volume initialization
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
