#!/bin/bash

# Define the persistent path for ChromaDB
DB_PATH="/data/chromadb"

# CHECK FOR PERSISTENCE: Only run ingestion if the database directory is empty
if [ -z "$(ls -A $DB_PATH)" ]; then
    echo "Database directory is empty. Running ingestion..."
    # The 'rm -rf' line is removed to allow persistence.
    python3 ingest.py
else
    echo "Database files found. Skipping ingestion."
fi

echo 'Starting the ChromaDB server...'
# *** IMPORTANT: Use the /data path for persistence ***
chroma run --path $DB_PATH --host 0.0.0.0 --port 8000 &

echo "Waiting for ChromaDB server to start..."
# Note: A simple sleep is often okay for simple deployments, but brittle.
sleep 10

echo "Starting the Streamlit app..."

streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false