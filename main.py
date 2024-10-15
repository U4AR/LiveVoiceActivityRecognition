import torch
import pyaudio
import numpy as np

# Set the number of threads for Torch (for performance tuning)
torch.set_num_threads(1)

# Load Silero VAD model and utilities
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
(get_speech_timestamps, _, _, _, _) = utils  # Extract the needed utility functions

# Constants for audio input
RATE = 16000  # Sampling rate (expected by the model)
CHUNK_DURATION_SEC = 0.5  # Chunk duration of 0.5 seconds (500 ms)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_SEC)  # Size of each chunk in samples

# Initialize PyAudio for microphone input
p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("Listening for voice activity...")

def normalize_audio(audio):
    """ Normalize audio to [-1, 1] range for the Silero model """
    return audio / np.max(np.abs(audio))

try:
    while True:
        # Read a chunk of audio from the microphone (0.5 second chunk)
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Convert audio data to a numpy array (16-bit PCM to float32)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Normalize the audio
        audio_np = normalize_audio(audio_np)

        # Convert the normalized audio to a tensor and add a batch dimension
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        # Perform VAD detection on the audio chunk
        speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=RATE, return_seconds=True)

        # Check if speech is detected
        if len(speech_timestamps) > 0:
            print("Speech detected!")
        else:
            print("Silence...")

except KeyboardInterrupt:
    # Cleanup when done
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stopped recording.")
