import streamlit as st
import torch
import io
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# Load model (cached for performance)
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

# Transcribe function - works with MP3, WAV, M4A, etc.
def transcribe_audio(audio_bytes, processor, model):
    # Save bytes to temporary file-like object
    with io.BytesIO(audio_bytes) as buf:
        # Load audio using librosa (handles MP3, WAV, etc.)
        audio_array, sample_rate = librosa.load(buf, sr=16000, mono=True)
    
    # Process audio for Wav2Vec2
    input_values = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Main app
def main():
    st.title("UNIQLO Fashion Review Analyzer")
    st.write("Upload an audio file to transcribe and analyze sentiment.")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        processor, model = load_model()
        
        with st.spinner("Transcribing..."):
            audio_bytes = uploaded_file.read()
            transcript = transcribe_audio(audio_bytes, processor, model)
        
        st.write("**Audio Transcript:**", transcript)
        
        # Simple sentiment analysis (placeholder - replace with your model)
        if "love" in transcript.lower() or "great" in transcript.lower() or "perfect" in transcript.lower():
            sentiment = "recommended"
        else:
            sentiment = "not recommended"
        
        st.write("**Sentiment Result:**", sentiment)

if __name__ == "__main__":
    main()
