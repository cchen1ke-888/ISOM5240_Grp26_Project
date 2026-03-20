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
    
    # Pipeline 2: Sentiment Analysis
    sentiment_pipeline = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    return processor, asr_model, sentiment_pipeline

# ============================================
# TRANSCRIPTION FUNCTION (WAV ONLY)
# ============================================
def transcribe_audio(audio_bytes, processor, model):
    """Convert uploaded WAV audio to text"""
    try:
        # Use soundfile for WAV files (works reliably)
        import soundfile as sf
        
        # Read WAV file from bytes
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
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
            <p><strong>📌 Instructions:</strong> Upload a <strong>WAV audio file</strong> containing a fashion review.</p>
            <p>The system will transcribe the audio and analyze the sentiment.</p>
            <p><em>Note: For best results, use WAV format. MP3 support coming soon.</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader - WAV only for now
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        help="Upload WAV files only for now"
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
        
        # Check if transcription failed
        if transcript.startswith("[Error:"):
            st.error(transcript)
            st.stop()
        
        st.markdown("**📝 Audio Transcript:**")
        st.info(transcript)
        
        # ============================================
        # PIPELINE 2: Sentiment Analysis
        # ============================================
        st.markdown("---")
        st.markdown("### 💭 Step 2: Analyzing Sentiment")
        
        with st.spinner("Analyzing..."):
            sentiment, confidence = analyze_sentiment(transcript, sentiment_pipeline)
        
        # Check if sentiment analysis failed
        if sentiment.startswith("[Error:"):
            st.error(sentiment)
            st.stop()
        
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
