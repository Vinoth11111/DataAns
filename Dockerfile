FROM python:3.11-slim

WORKDIR /app

# 1. Install System Tools
#RUN apt-get update && apt-get install -y \
#    build-essential \
 #   curl \
  #  && rm -rf /var/lib/apt/lists/*
    
# Create user
RUN useradd -m -u 1000 user
USER user

# Env variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 

# Install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy App (This copies your python files AND the chromadb folder you uploaded)
COPY --chown=user:user . . 

# 2. Permissions Setup (Root Sandwich)
# We ensure the user owns the pre-made database folder
#USER root
# We don't need to mkdir because COPY already put the folder there.
# We just need to make sure the user owns it.
#RUN chown -R user:user /app/chromadb
#RUN chmod -R 777 /app/chromadb
#USER user

EXPOSE 7860

# 3. THE INSTANT START SCRIPT
# - Removed 'rm -rf' (So we keep the data)
# - Removed 'ingest.py' (Data is already there)
# - Starts Chroma and Streamlit immediately
#RUN printf "#!/bin/bash\n\
#echo 'Starting ChromaDB Server...'\n\
#chroma run --path /app/chromadb --host 0.0.0.0 --port 8000 &\n\
#echo 'Waiting 5s for DB to warm up...'\n\
#sleep 5\n\
#echo 'Starting Streamlit...'\n\
#streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false\n" > start.sh

# Make executable
#RUN chmod +x start.sh

# Run
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]