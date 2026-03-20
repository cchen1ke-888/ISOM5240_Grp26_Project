import streamlit as st
import torch
import soundfile as sf
import io
import numpy as np
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)

# ============================================
# LOAD MODELS WITH CACHING
# ============================================
@st.cache_resource
def load_models():
    """Load all models with caching for better performance"""
    
    # Pipeline 1: Audio-to-Text (Wav2Vec2-Base)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model.eval()
    
    # Pipeline 2: Sentiment Analysis (Your fine-tuned model)
    # Replace with your actual fine-tuned model path on Hugging Face
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        "your-username/your-finetuned-model-name"  # CHANGE THIS!
    )
    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        "your-username/your-finetuned-model-name"  # CHANGE THIS!
    )
    
    # Alternative: Use pipeline for simplicity
    # sentiment_pipeline = pipeline(
    #     "text-classification",
    #     model="your-username/your-finetuned-model-name"
    # )
    
    return processor, asr_model, sentiment_tokenizer, sentiment_model

# ============================================
# TRANSCRIPTION FUNCTION
# ============================================
def transcribe_audio(audio_bytes, processor, model):
    """Convert uploaded audio to text using Wav2Vec2-Base"""
    
    # Read audio from bytes
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Ensure mono audio (convert to mono if stereo)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Process audio
    input_values = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# ============================================
# SENTIMENT ANALYSIS FUNCTION
# ============================================
def analyze_sentiment(text, tokenizer, model):
    """Classify text sentiment using fine-tuned model"""
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    # Map label to output
    # Assuming: label=1 = recommended (positive), label=0 = not recommended (negative)
    if predicted_class == 1:
        sentiment_result = "recommended"
    else:
        sentiment_result = "not recommended"
    
    return sentiment_result, confidence

# ============================================
# MAIN APP
# ============================================
def main():
    # ============================================
    # PAGE CONFIGURATION
    # ============================================
    st.set_page_config(
        page_title="UNIQLO Fashion Review Sentiment Analyzer",
        page_icon="👗",
        layout="wide"
    )
    
    # ============================================
    # TITLE (BOLD, LARGE)
    # ============================================
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 48px; font-weight: bold; color: #2c3e50;'>
            👗 UNIQLO Women's Fashion Review Analyzer
        </h1>
        <h3 style='text-align: center; font-size: 20px; color: #7f8c8d; margin-bottom: 30px;'>
            Audio-to-Text Transcription + Sentiment Analysis
        </h3>
        """,
        unsafe_allow_html=True
    )
    
    # ============================================
    # INSTRUCTIONS
    # ============================================
    st.markdown(
        """
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 16px; margin: 0;'>
                📌 <strong>Instructions:</strong> Upload an audio file containing a fashion review 
                (MP3, WAV, or M4A format). The system will transcribe the audio and analyze the sentiment 
                to determine if the reviewer recommends the product or not.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ============================================
    # FILE UPLOADER
    # ============================================
    uploaded_file = st.file_uploader(
        "Choose an audio file...",
        type=["mp3", "wav", "m4a", "flac"],
        help="Upload audio files in MP3, WAV, M4A, or FLAC format"
    )
    
    # ============================================
    # PROCESS UPLOADED FILE
    # ============================================
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📁 **File:** {uploaded_file.name}")
        with col2:
            file_size = uploaded_file.size / 1024  # KB
            st.info(f"📏 **Size:** {file_size:.1f} KB")
        with col3:
            file_type = uploaded_file.type.split('/')[-1].upper()
            st.info(f"🎵 **Format:** {file_type}")
        
        # Add a separator
        st.markdown("---")
        
        # Load models (cached)
        with st.spinner("🔄 Loading models... This may take a moment on first run."):
            processor, asr_model, sentiment_tokenizer, sentiment_model = load_models()
        
        # Read audio bytes
        audio_bytes = uploaded_file.read()
        
        # ============================================
        # PIPELINE 1: Audio-to-Text Transcription
        # ============================================
        st.markdown("### 🎙️ Step 1: Transcribing Audio...")
        
        with st.spinner("🎤 Transcribing your audio file..."):
            try:
                transcript = transcribe_audio(audio_bytes, processor, asr_model)
                st.success("✅ Transcription complete!")
            except Exception as e:
                st.error(f"❌ Error during transcription: {str(e)}")
                st.stop()
        
        # Display transcript
        st.markdown("---")
        st.markdown("### 📝 Audio Transcript")
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db;'>
                <p style='font-size: 16px; margin: 0;'><strong>Audio Transcript:</strong> {transcript}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ============================================
        # PIPELINE 2: Sentiment Analysis
        # ============================================
        st.markdown("### 💭 Step 2: Analyzing Sentiment...")
        
        with st.spinner("🔍 Analyzing sentiment..."):
            try:
                sentiment_result, confidence = analyze_sentiment(
                    transcript, sentiment_tokenizer, sentiment_model
                )
                st.success("✅ Sentiment analysis complete!")
            except Exception as e:
                st.error(f"❌ Error during sentiment analysis: {str(e)}")
                st.stop()
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Display transcript again with formatting
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <p style='font-size: 16px; margin: 0;'><strong>🎙️ Audio Transcript:</strong> {transcript}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display sentiment result with color coding
        if sentiment_result == "recommended":
            result_color = "#27ae60"  # Green
            result_emoji = "✅"
        else:
            result_color = "#e74c3c"  # Red
            result_emoji = "❌"
        
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid {result_color};'>
                <p style='font-size: 18px; margin: 0; font-weight: bold;'>
                    {result_emoji} <strong>Sentiment Result:</strong> {sentiment_result}
                </p>
                <p style='font-size: 14px; margin: 5px 0 0 0; color: #7f8c8d;'>
                    Confidence: {confidence:.2%}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ============================================
        # FOOTER / ADDITIONAL INFO
        # ============================================
        st.markdown("---")
        st.caption(
            "🚀 Powered by Hugging Face Transformers | "
            "Pipeline 1: Wav2Vec2-Base for Audio Transcription | "
            "Pipeline 2: Fine-tuned Sentiment Analysis Model"
        )
    
    else:
        # Show placeholder when no file uploaded
        st.markdown(
            """
            <div style='text-align: center; padding: 50px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;'>
                <p style='font-size: 18px; color: #7f8c8d;'>
                    📤 Upload an audio file above to begin analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()
