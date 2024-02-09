from typing import Union
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import array as arr

from fastapi import FastAPI, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
import starlette.status as status

app = FastAPI()

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def predict(audio_data : arr.array, sampling_rate : int):
    input_values = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values

    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

# Redirect to /docs (relative URL)
@app.get("/")
async def redirect_docs():
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)

# UploadFile is a wrapper for SpooledTemporaryFile, and is expected to be destroyed as soon as the file is closed
# See https://docs.python.org/3/library/tempfile.html#tempfile.SpooledTemporaryFile for more details
@app.post("/asr")
async def transcribe_audio(file: UploadFile):
    audio_data, sampling_rate = sf.read(file.file)

    transcription = predict(audio_data, sampling_rate)

    return {
        "transcription" : transcription,
        "duration" : "{:.1f}".format(len(audio_data) / sampling_rate)
    }

# API to return pong on ping GET request
@app.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "pong"
