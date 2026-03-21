import streamlit as st
from transformers import pipeline
import librosa
import tempfile
import os
from datetime import datetime

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
        layout="wide",
        initial_sidebar_state="collapsed")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .upload-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .result-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Large bold title with styling
    st.markdown("""
        <div class="main-header">
            <h1 style='font-weight: bold; color: #1f3e5c;'>👗 UNIQLO User Review Sentiment Analyzer</h1>
            <p style='color: #666;'>Transform audio reviews into actionable insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Upload section with card-like styling
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload Audio File")
        st.write("Upload a user review audio file to analyze sentiment")
        
        # File uploader with custom styling
        audio_file = st.file_uploader(
            "Choose an audio file (MP3 or WAV format)",
            type=['mp3', 'wav'],
            help="Supported formats: MP3, WAV. File size should be under 200MB"
        )
        
        # Display file info if uploaded
        if audio_file:
            file_size = audio_file.size / 1024 / 1024  # Convert to MB
            st.info(f"📁 **File:** {audio_file.name} ({file_size:.1f} MB)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main processing area
    if audio_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Transcription
            status_text.text("🎤 Step 1/2: Transcribing audio...")
            progress_bar.progress(25)
            
            transcript = transcribe_audio(tmp_path)
            progress_bar.progress(50)
            
            # Display transcript in a nice card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📝 Audio Transcript")
            st.markdown(f'<p style="font-size: 16px; line-height: 1.6;">{transcript}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 2: Sentiment Analysis
            status_text.text("💭 Step 2/2: Analyzing sentiment...")
            progress_bar.progress(75)
            
            sentiment_result = analyze_sentiment(transcript)
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display sentiment results
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']
            
            # Map label to recommended/not recommended
            if sentiment_label == "1" or sentiment_label.upper() == "POSITIVE":
                recommendation = "Recommended"
                emoji = "👍"
                sentiment_color = "green"
                bg_color = "#d4edda"
                border_color = "#c3e6cb"
            elif sentiment_label == "0" or sentiment_label.upper() == "NEGATIVE":
                recommendation = "Not Recommended"
                emoji = "👎"
                sentiment_color = "red"
                bg_color = "#f8d7da"
                border_color = "#f5c6cb"
            else:
                recommendation = sentiment_label
                emoji = "😐"
                sentiment_color = "orange"
                bg_color = "#fff3cd"
                border_color = "#ffeeba"
            
            # Display sentiment result card
            st.markdown(f"""
                <div class="result-card" style="background-color: {bg_color}; border-left: 4px solid {border_color};">
                    <h3 style="color: {sentiment_color}; margin-bottom: 1rem;">🎯 Sentiment Analysis Result</h3>
                    <p style="font-size: 24px; font-weight: bold; margin-bottom: 0.5rem;">{recommendation} {emoji}</p>
                    <p style="font-size: 14px; color: #666;">Confidence Score: {sentiment_score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add confidence meter
            st.markdown("### 📊 Confidence Meter")
            st.progress(sentiment_score, text=f"{sentiment_score:.0%} confidence")
            
            # Add summary section
            st.markdown("---")
            st.markdown("### 📋 Analysis Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.metric("File Processed", audio_file.name.split('.')[0][:20] + "...")
                st.metric("Audio Length", f"{librosa.get_duration(path=tmp_path):.1f} seconds")
            with summary_col2:
                st.metric("Transcript Length", f"{len(transcript.split())} words")
                st.metric("Processing Time", "Complete")
            
            # Add download button for transcript
            st.download_button(
                label="📥 Download Transcript",
                data=transcript,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Download the transcribed text as a text file"
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            st.info("Please try uploading a different audio file or check if the file format is supported.")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        # Show placeholder when no file is uploaded
        st.markdown('<div style="text-align: center; padding: 3rem; color: #666;">', unsafe_allow_html=True)
        st.markdown("### 🎧 Ready to analyze user reviews")
        st.markdown("Upload an audio file to get started")
        st.markdown("---")
        st.markdown("#### Supported formats: MP3, WAV")
        st.markdown("#### Features:")
        st.markdown("- 🎤 Automatic speech transcription")
        st.markdown("- 💭 Sentiment analysis using fine-tuned model")
        st.markdown("- 📊 Confidence scoring")
        st.markdown("- 📥 Download transcripts")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 12px;'>Powered by Hugging Face Transformers | UNIQLO Sentiment Analysis Model</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
