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

    # Convert to lowercase and capitalize the first letter of each sentence
    text = result['text'].lower()
    # Capitalize first letter of the entire text
    text = text[0].upper() + text[1:] if text else text
    # Add period after text
    if text and not text.endswith('.'):
        text = text + '.'
    
    return text

def analyze_sentiment(text):
    # Load the sentiment analysis pipeline with the fine-tuned model I uploaded to Hugging Face.
    sentiment_pipeline = pipeline("sentiment-analysis", model="cykChloe/ISOM5240-Grp26-Sentiment")
    
    # Analyze sentiment
    result = sentiment_pipeline(text)
    
    return result[0]

def main():
    # Page Configuration
    st.set_page_config(
        page_title="UNIQLO User Review Sentiment Analyzer",
        page_icon="👗",
        layout="wide")
    
    # Custom CSS for styling without background colors
    st.markdown("""
        <style>
        .transcript-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #1f3e5c;
        }
        .sentiment-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Large bold title
    st.markdown("<h1 style='font-weight: bold;'>👗 UNIQLO User Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
    
    # Simple instructions
    st.write("Please upload an audio file of the user review about UNIQLO you collected from social media:")
    
    # Upload the audio file of a user review
    audio_file = st.file_uploader("", type=['mp3', 'wav'])

    # Step 1: Transcribe an audio file into plain text.
    if audio_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Transcribe by calling the function
            transcript = transcribe_audio(tmp_path)
            
            # Display the transcript in styled card
            st.markdown('<div class="transcript-card">', unsafe_allow_html=True)
            st.subheader("📝 Audio Transcript")
            st.write(f"{transcript}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 2: Perform sentiment analysis on the transcribed text using the fine-tuned model.
            sentiment_result = analyze_sentiment(transcript)
                            
            # Display sentiment results
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']
                
            # Map label to recommended/not recommended
            if sentiment_label == "1" or sentiment_label.upper() == "POSITIVE":
                recommendation = "Recommended"
                emoji = "👍"
                sentiment_color = "green"
            elif sentiment_label == "0" or sentiment_label.upper() == "NEGATIVE":
                recommendation = "Not Recommended"
                emoji = "👎"
                sentiment_color = "red"
            else:
                recommendation = sentiment_label
                emoji = "😐"
                sentiment_color = "orange"
            
            # Display sentiment result in styled card
            st.markdown('<div class="sentiment-card">', unsafe_allow_html=True)
            st.subheader("🎯 Sentiment Analysis Result")
            st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {sentiment_color}; margin-bottom: 0.5rem;'>{recommendation} {emoji}</p>", unsafe_allow_html=True)
            st.write(f"**Confidence Score:** {sentiment_score:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
if __name__ == "__main__":
    main()
