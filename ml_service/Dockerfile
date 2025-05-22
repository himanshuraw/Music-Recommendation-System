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

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies including gosu
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 gosu && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY main.py .
COPY entrypoint.sh .

# Copy initial data and models (these will be used to initialize volumes)
COPY data/ /app/build_data/
COPY models/ /app/build_models/

# Declare mount points
VOLUME ["/app/data", "/app/models"]

# Create directories (ownership handled in entrypoint)
RUN mkdir -p /app/data /app/models

# Set permissions and add non-root user
RUN chmod +x /app/entrypoint.sh && \
    adduser --disabled-password --gecos "" appuser

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

# Use entrypoint script to handle setup
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]