FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./asr/requirements.txt /code/asr/requirements.txt

# Installs all required python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/asr/requirements.txt

# Copies the API script
COPY ./asr/asr_api.py /code/asr/asr_api.py

# Installs git LFS for cloning large files
RUN apt-get update && \
    apt-get install -y git-lfs

# Clones the model for the web server to use
RUN git clone https://huggingface.co/facebook/wav2vec2-large-960h asr/wav2vec2-large-960h

CMD ["uvicorn", "asr.asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
