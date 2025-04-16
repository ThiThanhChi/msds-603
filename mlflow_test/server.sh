#!/bin/bash

# mlflow db upgrade $POSTGRESQL_URL
# mlflow server \
#   --host 0.0.0.0 \
#   --port 8080 \
#   --backend-store-uri $POSTGRESQL_URL \
#   --artifacts-destination $STORAGE_URL


#!/bin/bash
# Run database migrations (optional but good)
mlflow db upgrade $POSTGRESQL_URL
# Start MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port ${PORT:-8080} \
  --backend-store-uri $POSTGRESQL_URL \
  --default-artifact-root $STORAGE_URL
