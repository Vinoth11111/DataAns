FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
# create a non root user
RUN useradd -m -u 1000 user

#swithch to non root user
USER user

# create vertual environment
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 
# PYTHONUNBUFFERED=1 IS USED FOR REAL TIME LOGGING. BY DEFAULT THE LOGS WE SEND FLUSH WAY ALL LOGS STORE AND SEDN AT A TIME(BUFFRED). THIS WILL MAKE SURE ALL LOGS ARE SENT IN REAL TIME.
#COPY requirements.txt .

#chown means CHangeOWNership

COPY --chown=user:user requirements.txt .
# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the app files to the container

COPY --chown=user:user . . 
USER root

#RUN dos2unix start.sh && chmod +x start.sh
#COPY . .
# create a makeout path for seeing the files in chromadb
# below line is added to provide access to non root user in docker container.
# mkdir makes make directory -p means make PARENT directories as needed.

RUN mkdir -p /app/chromadb
# provide access to the non root user in docker container.


RUN chown -R user:user /app/chromadb

# changemode 777 - read write execute for all users
RUN chmod -R 777 /app/chromadb

# EXPOSE THE PORT,7860 is for huggingface space streamlit apps and 8501 is for normal streamlit apps
EXPOSE 7860
#EXPOSE 8501

#RUN <<EOF cat > /app/start.sh
#echo 'starting the chromadb server'
#chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &
#echo 'wait for 5 seconds to let chromadb start'
#sleep 5 &&
#echo 'starting the data ingestion'
#python3 ingest.py &
#echo 'starting the streamlit app'
#streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false
#EOF

RUN printf "!#/bin/bash\n \
echo 'starting the chromadb server'\n \
chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &\n \
echo 'wait for 5 seconds to let chromadb start'\n \
sleep 5 && \n \
echo 'starting the data ingestion'\n \
(sleep 20 && python3 ingest.py) & \n \
echo 'starting the streamlit app'\n \
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false" > start.sh

RUN chmod +x ./start.sh

USER user

CMD [ "bash", "./start.sh" ]
# run the app
#/bin/bash run bash shell and -c means run the command inside the quotes.
#CMD ["/bin/bash","-c","chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 & python3 ingest.py & streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false"]



