import streamlit as st
import torch
import io
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import warnings
warnings.filterwarnings("ignore")

# Load model (cached for performance)
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

# Transcribe function - uses librosa ONLY
def transcribe_audio(audio_bytes, processor, model):
    try:
        # Use librosa to load audio directly from bytes
        # librosa.load can handle MP3, WAV, M4A, etc.
        audio_array, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=16000,           # Resample to 16kHz for Wav2Vec2
            mono=True           # Convert to mono
        )
        
        # Process audio
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
        
    except Exception as e:
        return f"Error: {str(e)}"

# Main app
def main():
    st.set_page_config(page_title="UNIQLO Review Analyzer", page_icon="👗")
    
    # Title
    st.markdown("# 👗 UNIQLO Fashion Review Analyzer")
    st.markdown("Upload an audio file to transcribe and analyze sentiment.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload MP3, WAV, M4A, FLAC, or OGG files"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Load models
        with st.spinner("Loading models..."):
            processor, model = load_model()
        
        # Transcribe
        with st.spinner("Transcribing audio..."):
            audio_bytes = uploaded_file.read()
            transcript = transcribe_audio(audio_bytes, processor, model)
        
        # Display transcript
        st.markdown("---")
        st.markdown("### 📝 Audio Transcript")
        st.info(transcript)
        
        # Simple sentiment analysis (replace with your fine-tuned model later)
        positive_words = ["love", "great", "perfect", "amazing", "beautiful", "good", "nice", "recommend"]
        negative_words = ["bad", "poor", "terrible", "awful", "disappointed", "hate", "worst"]
        
        transcript_lower = transcript.lower()
        
        if any(word in transcript_lower for word in positive_words):
            sentiment = "✅ recommended"
        elif any(word in transcript_lower for word in negative_words):
            sentiment = "❌ not recommended"
        else:
            sentiment = "⚖️ neutral"
        
        st.markdown("### 💭 Sentiment Result")
        st.success(f"**{sentiment}**")

if __name__ == "__main__":
    main()
