echo 'starting the chromadb server'
chromadb run --host chromadb_server --port 8000 &

echo "waiting for chromadb server to start"
sleep 10

echo "running the ingestion file"
python3 ingest.py

echo "starting the streamlit app"
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
