import streamlit as st
from transformers import pipeline
import librosa
import tempfile
import os

def transcribe_audio(audio_path):
    # Load audio and resample to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Load the transcriber pipeline
    transcriber = pipeline("automatic-speech-recognition", 
                          model="facebook/wav2vec2-base-960h")
    
    # Transcribe
    result = transcriber(audio)
    
    return result['text']

def main():
    # Page Configuration
    st.set_page_config(
        page_title="UNIQLO User Review Sentiment Analyzer",
        page_icon="👗",
        layout="wide")
    
    # Large bold title
    st.markdown("<h1 style='font-weight: bold;'>UNIQLO User Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
    
    # Simple instructions
    st.write("Please upload the audio file of a user review:")
    
    # Upload the audio file of a user review
    audio_file = st.file_uploader("", type=['mp3', 'm4a', 'wav'])
    
    if audio_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Transcribe using the function
            transcript = transcribe_audio(tmp_path)
            
            # Display the transcript
            st.write(f"**Audio Transcript:** {transcript}")
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
