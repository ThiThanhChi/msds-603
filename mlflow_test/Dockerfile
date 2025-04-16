# FROM python:3.10-slim

# WORKDIR /app

# RUN pip install --no-cache-dir mlflow

# EXPOSE 5000

# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

FROM python:3.12-slim

WORKDIR /

COPY requirements.txt requirements.txt
COPY server.sh server.sh

ENV GOOGLE_APPLICATION_CREDENTIALS='./secrets/credentials'

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

RUN chmod +x server.sh

ENTRYPOINT ["./server.sh"]
