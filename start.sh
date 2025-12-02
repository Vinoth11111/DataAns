echo "Cleaning up database files..."
rm -rf /app/chromadb/*

echo 'starting the chromadb server'
chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &

echo "waiting for chromadb server to start"
sleep 10

echo "running the ingestion file"
python3 ingest.py

