FROM python:3.9.5-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    mkdir -p /app
    
WORKDIR /app/

COPY requirements.txt /tmp/
COPY setup.py .
COPY src src

RUN pip install -r /tmp/requirements.txt && \
    python setup.py install

WORKDIR /app/src/scripts
ENTRYPOINT /bin/bash