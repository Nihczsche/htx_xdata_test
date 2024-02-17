from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import snapshot_download
import soundfile as sf
import torch
import array as arr
import librosa
import os

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
import starlette.status as status

app = FastAPI()

MODEL_NAME = "wav2vec2-large-960h"
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), MODEL_NAME)

pretrained_model_name_or_path = "facebook/wav2vec2-large-960h"
if os.path.exists(MODEL_PATH):
    pretrained_model_name_or_path = MODEL_PATH

# Load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path)

def predict(audio_data : arr.array, sampling_rate : int):
    """
    Performs the transcription of the audio input

    [in] audio_data : Audio data in an array format
    [in] sampling_rate : The sampling rate of the audio data

    [out] transcription : Return the transcribed string
    """

    if sampling_rate != processor.feature_extractor.sampling_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=processor.feature_extractor.sampling_rate)

    input_values = processor(audio_data, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding="longest").input_values

    with torch.no_grad():
        logits = model(input_values.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

@app.get("/")
def redirect_docs():
    """
    Redirects any requests for the main page to the docs page

    [out] RedirectResponse
    """
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)

# UploadFile is a wrapper for SpooledTemporaryFile, and is expected to be destroyed as soon as the file is closed
# See https://docs.python.org/3/library/tempfile.html#tempfile.SpooledTemporaryFile for more details
@app.post("/asr")
def transcribe_audio(file: UploadFile):
    """
    Reads an uploaded audio file into an array type, transcribes then returns the text
    as a Json containing the transcribed string and audio file duration

    [out] JsonResponse containing transcribed text and audio data duration
    """

    try:
        audio_data, sampling_rate = sf.read(file.file)
    except Exception as e:
        return HTTPException(status_code=404, 
                             detail="Could not read audio file: {}".format(str(e)))

    transcription = predict(audio_data, sampling_rate)

    return {
        "transcription" : transcription,
        "duration" : "{:.1f}".format(len(audio_data) / sampling_rate)
    }

@app.get("/ping", response_class=PlainTextResponse)
def ping():
    """
    API to return pong on ping GET request

    [out] PlainTextResponse "pong"
    """

    return "pong"
