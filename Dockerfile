# FROM python:3.12-slim

# WORKDIR /

# COPY requirements.txt requirements.txt
# COPY server.sh server.sh

# ENV GOOGLE_APPLICATION_CREDENTIALS='./secrets/credentials'

# RUN pip install --upgrade pip && pip install -r requirements.txt

# EXPOSE 8080

# RUN chmod +x server.sh

# ENTRYPOINT ["./server.sh"]


FROM python:3.10-slim  # Changed to 3.10 to match your flow

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install metaflow==2.15.9  # Explicitly add Metaflow

# Add your credentials (be careful with this in production)
COPY secrets/credentials ./secrets/credentials

# Add your server script if needed
COPY server.sh .

# For Metaflow Kubernetes runs
ENV METAFLOW_DEFAULT_DATASTORE=kubernetes
ENV METAFLOW_KUBERNETES_NAMESPACE=default
ENV GOOGLE_APPLICATION_CREDENTIALS='/app/secrets/credentials'

# Set entrypoint for both Metaflow and your server
ENTRYPOINT ["/bin/bash"]