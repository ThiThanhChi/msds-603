# FROM python:3.13-slim

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PORT=8000

# WORKDIR /app

# # Install system dependencies first (if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -U pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy application files
# COPY reddit_model_pipeline.joblib redditApp.py ./

# # Run as non-root user for security
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# # Health check (optional but recommended)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:$PORT/docs || exit 1

# EXPOSE $PORT
# CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY reddit_model_pipeline.joblib redditApp.py ./

EXPOSE 8000
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]