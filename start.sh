echo "Cleaning up database files..."
rm -rf /app/chromadb/*

echo 'starting the chromadb server'
chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &

echo "waiting for chromadb server to start"
sleep 10

echo "running the ingestion file"
python3 ingest.py

echo "starting the streamlit app"
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false
