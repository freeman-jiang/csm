import os

import torch
import torchaudio
from generator import Generator, Segment, load_csm_1b

# Disable Triton/TorchInductor compilation
os.environ["NO_TORCH_COMPILE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device)
print(f"Using device: {device}")

speakers = [0, 0, 0, 0]
transcripts = [
    "1 2 3 4 5 6 7 8 9 10",
    "Hey do you care about the story of how many hours the artist has to put in?",
    "I think that's generally a good take",
    "It just became worse",
]
audio_paths = [
    "wav/1 2 3 4 5 6 7 8 9 10.wav",
    "wav/Hey do you care about the story of how many hours the artist has to put in.wav",
    "wav/I think that's generally a good take.wav",
    "wav/It just became worse.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="This approach works for basic voice cloning, but it's not as robust as actual fine tuning. For each new generation, you'll need to provide these samples again as context.",
    speaker=0,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)