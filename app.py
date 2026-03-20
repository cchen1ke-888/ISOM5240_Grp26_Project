import streamlit as st
import torch
import io
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load all models with caching"""
    
    # Pipeline 1: Audio-to-Text (Wav2Vec2-Base)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model.eval()
    
    # Pipeline 2: Sentiment Analysis - Using a simple pipeline for now
    # REPLACE THIS WITH YOUR FINE-TUNED MODEL LATER
    sentiment_pipeline = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    return processor, asr_model, sentiment_pipeline

# ============================================
# TRANSCRIPTION FUNCTION
# ============================================
def transcribe_audio(audio_bytes, processor, model):
    """Convert uploaded audio to text"""
    try:
        # Use librosa to load audio
        import librosa
        audio_array, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=16000,
            mono=True
        )
        
        # Process audio
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
        
    except Exception as e:
        return f"[Error: {str(e)}]"

# ============================================
# SENTIMENT ANALYSIS FUNCTION
# ============================================
def analyze_sentiment(text, sentiment_pipeline):
    """Analyze sentiment using the pipeline"""
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Convert to recommended/not recommended format
        if label == "POSITIVE":
            return "recommended", score
        else:
            return "not recommended", score
            
    except Exception as e:
        return f"[Error: {str(e)}]", 0.0

# ============================================
# MAIN APP
# ============================================
def main():
    st.set_page_config(page_title="UNIQLO Review Analyzer", page_icon="👗")
    
    # Title
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 48px; font-weight: bold;'>
            👗 UNIQLO Fashion Review Analyzer
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Instructions
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <p><strong>📌 Instructions:</strong> Upload an audio file containing a fashion review.</p>
            <p>The system will transcribe the audio and analyze the sentiment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac"],
        help="Upload MP3, WAV, M4A, or FLAC files"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Load models
        with st.spinner("🔄 Loading models..."):
            processor, asr_model, sentiment_pipeline = load_models()
        
        # ============================================
        # PIPELINE 1: Transcription
        # ============================================
        st.markdown("---")
        st.markdown("### 🎙️ Step 1: Transcribing Audio")
        
        with st.spinner("Transcribing..."):
            audio_bytes = uploaded_file.read()
            transcript = transcribe_audio(audio_bytes, processor, asr_model)
        
        st.markdown("**📝 Audio Transcript:**")
        st.info(transcript)
        
        # ============================================
        # PIPELINE 2: Sentiment Analysis
        # ============================================
        st.markdown("---")
        st.markdown("### 💭 Step 2: Analyzing Sentiment")
        
        with st.spinner("Analyzing..."):
            sentiment, confidence = analyze_sentiment(transcript, sentiment_pipeline)
        
        # Display result with color coding
        if sentiment == "recommended":
            st.markdown(
                f"""
                <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745;'>
                    <p style='font-size: 18px; margin: 0; font-weight: bold;'>✅ Sentiment Result: {sentiment}</p>
                    <p style='font-size: 14px; margin: 5px 0 0 0; color: #6c757d;'>Confidence: {confidence:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545;'>
                    <p style='font-size: 18px; margin: 0; font-weight: bold;'>❌ Sentiment Result: {sentiment}</p>
                    <p style='font-size: 14px; margin: 5px 0 0 0; color: #6c757d;'>Confidence: {confidence:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.caption("🚀 Powered by Wav2Vec2-Base for transcription | DistilBERT for sentiment analysis")

if __name__ == "__main__":
    main()
