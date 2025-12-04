#!/bin/bash

echo 'Starting the ChromaDB server...'
# *** IMPORTANT: Use the /data path for persistence ***
chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &

echo "Waiting for ChromaDB server to start..."
# Note: A simple sleep is often okay for simple deployments, but brittle.
sleep 10

# Run the ingestion script in the background
echo "Running ingestion..."
    # The 'rm -rf' line is removed to allow persistence.
python3 ingest.py &





echo "Starting the Streamlit app..."

streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false