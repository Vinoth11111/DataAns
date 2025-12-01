echo "running the ingestion file"
python3 ingest.py

echo "starting the streamlit app"
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
