import streamlit as st
import torch
import io
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline
from pydub import AudioSegment
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
    
    # Pipeline 2: Sentiment Analysis (placeholder - replace with your fine-tuned model)
    sentiment_pipeline = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    return processor, asr_model, sentiment_pipeline

# ============================================
# AUDIO TO NUMPY FUNCTION (Supports MP3, M4A, WAV)
# ============================================
def audio_to_numpy(audio_bytes):
    """Convert any audio format to 16kHz mono numpy array"""
    
    # Load audio using pydub (handles MP3, M4A, WAV, etc.)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Set sample rate to 16000 Hz (required by Wav2Vec2)
    audio = audio.set_frame_rate(16000)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalize to [-1, 1] range based on sample width
    max_value = 2 ** (audio.sample_width * 8 - 1)
    samples = samples / max_value
    
    return samples, 16000

# ============================================
# TRANSCRIPTION FUNCTION
# ============================================
def transcribe_audio(audio_bytes, processor, model):
    """Convert uploaded audio to text using Wav2Vec2"""
    try:
        # Convert audio to numpy array
        audio_array, sample_rate = audio_to_numpy(audio_bytes)
        
        # Process audio
        input_values = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_values
        
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
        <h1 style='text-align: center; font-size: 48px; font-weight: bold; color: #2c3e50;'>
            👗 UNIQLO Fashion Review Analyzer
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Instructions
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
            <p style='font-size: 16px; margin: 0;'>
                📌 <strong>Instructions:</strong> Upload an audio file containing a fashion review.
                The system will transcribe the audio and analyze the sentiment.
            </p>
            <p style='font-size: 14px; margin: 10px 0 0 0; color: #666;'>
                ✅ Supported formats: MP3, M4A, WAV, FLAC, OGG
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader - supports multiple formats
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "m4a", "wav", "flac", "ogg"],
        help="Upload MP3, M4A, WAV, FLAC, or OGG files"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Show file info
        file_size = uploaded_file.size / 1024
        st.caption(f"File: {uploaded_file.name} | Size: {file_size:.1f} KB")
        
        # Load models
        with st.spinner("🔄 Loading models... (this may take a moment on first run)"):
            processor, asr_model, sentiment_pipeline = load_models()
        
        # ============================================
        # PIPELINE 1: Transcription
        # ============================================
        st.markdown("---")
        st.markdown("### 🎙️ Step 1: Transcribing Audio")
        
        with st.spinner("🎤 Transcribing... (this may take 10-20 seconds)"):
            audio_bytes = uploaded_file.read()
            transcript = transcribe_audio(audio_bytes, processor, asr_model)
        
        # Check if transcription failed
        if transcript.startswith("[Error:"):
            st.error(f"Transcription failed: {transcript}")
            st.info("💡 Tip: Try using a shorter audio clip or check the audio quality.")
            st.stop()
        
        st.markdown("**📝 Audio Transcript:**")
        st.success(transcript)
        
        # ============================================
        # PIPELINE 2: Sentiment Analysis
        # ============================================
        st.markdown("---")
        st.markdown("### 💭 Step 2: Analyzing Sentiment")
        
        with st.spinner("🔍 Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(transcript, sentiment_pipeline)
        
        # Check if sentiment analysis failed
        if sentiment.startswith("[Error:"):
            st.error(f"Sentiment analysis failed: {sentiment}")
            st.stop()
        
        # Display result with color coding
        if sentiment == "recommended":
            st.markdown(
                f"""
                <div style='background-color: #d4edda; padding: 18px; border-radius: 10px; border-left: 6px solid #28a745;'>
                    <p style='font-size: 20px; margin: 0; font-weight: bold; color: #155724;'>
                        ✅ Sentiment Result: {sentiment}
                    </p>
                    <p style='font-size: 14px; margin: 8px 0 0 0; color: #6c757d;'>
                        Confidence: {confidence:.2%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background-color: #f8d7da; padding: 18px; border-radius: 10px; border-left: 6px solid #dc3545;'>
                    <p style='font-size: 20px; margin: 0; font-weight: bold; color: #721c24;'>
                        ❌ Sentiment Result: {sentiment}
                    </p>
                    <p style='font-size: 14px; margin: 8px 0 0 0; color: #6c757d;'>
                        Confidence: {confidence:.2%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # ============================================
        # Footer
        # ============================================
        st.markdown("---")
        st.caption(
            "🚀 **Powered by:** Wav2Vec2-Base (Audio Transcription) | "
            "DistilBERT (Sentiment Analysis) | pydub + ffmpeg (Audio Processing)"
        )

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()
