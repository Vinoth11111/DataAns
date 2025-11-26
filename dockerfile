FROM python:3.11-slim

WORKDIR /app

# create a non root user
#RUN useradd -m -u 1000 user

#USER user
# create vertual environment
#ENV HOME=/home/user \
#    PATH=/home/user/local/bin:$PATH
COPY requirements.txt .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the app files to the container

COPY . .

# EXPOSE THE PORT,7860 is for huggingface space streamlit apps and 8501 is for normal streamlit apps
EXPOSE 7860

CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0"]