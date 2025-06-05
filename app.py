import os
import torch # type: ignore
from pytube import YouTube # type: ignore
from huggingsound import SpeechRecognitionModel # type: ignore
import librosa # type: ignore
import soundfile as sf # type: ignore
from transformers import pipeline # type: ignore

# Download YouTube video audio
VIDEO_URL = "https://www.youtube.com/watch?v=hWLf6JFbZoo"
yt = YouTube(VIDEO_URL)
yt.streams.filter(only_audio=True, file_extension='mp4').first().download(filename='ytaudio.mp4')

# Convert audio to WAV format
os.system('ffmpeg -i ytaudio.mp4 -acodec pcm_s16le -ar 16000 ytaudio.wav')

# Load speech recognition model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)

# Load audio file and split into 30-second chunks
input_file = 'ytaudio.wav'
print(librosa.get_samplerate(input_file))
stream = librosa.stream(input_file, block_length=30, frame_length=16000, hop_length=16000)

# Write each chunk to a separate WAV file
for i, speech in enumerate(stream):
    sf.write(f'{i}.wav', speech, 16000)

# Get the number of chunks
i += 1

# Create a list of audio file paths
audio_path = [f'{j}.wav' for j in range(i)]

# Transcribe each audio file
transcriptions = model.transcribe(audio_path)

# Combine transcriptions into a single string
full_transcript = ' '
for item in transcriptions:
    full_transcript += ''.join(item['transcription'])

# Summarize the transcript
summarization = pipeline('summarization')
summarized_text = summarization(full_transcript)[0]['summary_text']

print(summarized_text)