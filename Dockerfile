FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./asr /code/asr

RUN pip install --no-cache-dir --upgrade -r /code/asr/requirements.txt

CMD ["uvicorn", "asr.asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
