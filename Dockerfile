FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --chown=user:user requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY --chown=user:user . .
USER root

RUN mkdir -p ./db && \
    chown -R user:user ./db && \
    chmod 777 ./db

RUN printf "#!/bin/bash \n \
echo 'starting the chromadb server' \n \
chroma run --path ./db --host 0.0.0.0 --port 8000 && \n \
sleep 5 \n \
echo 'starting the ingestion process' \n \
python ingest.py & \n \
echo 'starting the main application' \n \
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableWebsocketCompression=false" > start.sh

RUN chmod +x ./start.sh

USER user

CMD [ "bash", "./start.sh" ]

