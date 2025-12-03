FROM python:3.11-slim-bookworm

WORKDIR /app

# 1. Install System Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
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

# Copy App
COPY --chown=user:user . . 

# 2. Permissions Setup
USER root
RUN mkdir -p /app/db
RUN chown -R user:user /app/db
RUN chmod -R 777 /app/db
EXPOSE 7860

# 3. CREATE PYTHON BOOT SCRIPT (The Robust Fix)
# We use Python's subprocess to guarantee Streamlit starts INSTANTLY
# while Ingestion runs quietly in the background.
RUN printf "import subprocess\n\
import time\n\
import os\n\
import threading\n\
\n\
def run_ingestion():\n\
    print('⏳ Ingestion delayed for 10 seconds...')\n\
    time.sleep(10)\n\
    print('▶️ Starting Ingestion in background...')\n\
    subprocess.run(['python3', 'ingest.py'])\n\
\n\
print('🧹 Cleaning Database...')\n\
os.system('rm -rf /app/db/*')\n\
\n\
print('💽 Starting db...')\n\
chroma_process = subprocess.Popen(['chroma', 'run', '--path', '/app/db', '--host', '0.0.0.0', '--port', '8000'])\n\
\n\
print('🚀 Starting Streamlit...')\n\
# Start Streamlit immediately so Hugging Face sees the port open\n\
streamlit_process = subprocess.Popen([\n\
    'streamlit', 'run', 'app.py',\n\
    '--server.port=7860',\n\
    '--server.address=0.0.0.0',\n\
    '--server.enableCORS=false',\n\
    '--server.enableXsrfProtection=false',\n\
    '--server.enableWebsocketCompression=false'\n\
])\n\
\n\
# Start ingestion in a separate thread so it doesn't block Streamlit\n\
threading.Thread(target=run_ingestion).start()\n\
\n\
# Keep the container running as long as Streamlit is alive\n\
streamlit_process.wait()\n" > boot.py

USER user


# Run the Python Boot Script
CMD ["python3", "boot.py"]