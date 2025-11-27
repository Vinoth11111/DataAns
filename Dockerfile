FROM python:3.11-slim

WORKDIR /app

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
#COPY . .
# create a makeout path for seeing the files in chromadb
# below line is added to provide access to non root user in docker container.
# mkdir makes make directory -p means make PARENT directories as needed.
#RUN mkdir -p /app/chromadb
# provide access to the non root user in docker container.

USER root

RUN chown -R user:user /app/chromadb

# changemode 777 - read write execute for all users
RUN chmod -R 777 /app/chromadb

USER user
# EXPOSE THE PORT,7860 is for huggingface space streamlit apps and 8501 is for normal streamlit apps
EXPOSE 7860
#EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0"]