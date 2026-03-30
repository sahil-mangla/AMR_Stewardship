# Dockerfile — Hospital ASP Coordinator Environment
# Builds a containerized OpenEnv environment for Hugging Face Spaces
#
# Build:  docker build -t asp-env .
# Run:    docker run -p 7860:7860 asp-env
# Test:   curl http://localhost:7860/health

FROM python:3.11-slim

# Metadata
LABEL maintainer="ASP Environment"
LABEL description="Hospital Antimicrobial Stewardship Program — OpenEnv Environment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY env/       ./env/
COPY tasks/     ./tasks/
COPY server.py  .
COPY inference.py .
COPY openenv.yaml .

# Create __init__.py files so Python treats dirs as packages
RUN touch env/__init__.py tasks/__init__.py

# HF Spaces requires port 7860
EXPOSE 7860

# Health check — judges ping this
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
