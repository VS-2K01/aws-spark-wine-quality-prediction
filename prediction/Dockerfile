  GNU nano 8.3                                                 Dockerfile
FROM python:3.10-slim

# Install Java (required by PySpark/Spark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy prediction script and trained model into the image
COPY winePrediction_application.py /app/
COPY winePrediction_model_ensemble /app/winePrediction_model_ensemble

# Install Python dependencies: PySpark, NumPy, Pandas
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    numpy \
    pandas

# Entrypoint script to accept testFilePath and call Python
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]