import streamlit as st
from transformers import pipeline
import librosa
import tempfile
import os

def main():
    # Large bold title
    st.markdown("<h1 style='font-weight: bold;'>Audio Transcription</h1>", unsafe_allow_html=True)
    
    # Simple instructions
    st.write("Please upload the audio file of a user review:")
    
    # Upload the audio file of a user review
    audio_file = st.file_uploader("", type=['mp3', 'm4a', 'wav'])
    
    if audio_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        
        # Load audio and resample to 16kHz
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        # Transcribe
        transcriber = pipeline("automatic-speech-recognition", 
                               model="facebook/wav2vec2-base-960h")
        result = transcriber(audio)
        
        # Display the transcript
        st.write(f"**Audio Transcript:** {result['text']}")
        
        # Clean up
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
