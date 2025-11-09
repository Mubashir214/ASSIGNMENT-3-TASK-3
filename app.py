import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import os
import time

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
        line-height: 1.6;
    }
    .stats-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #fffacd;
        padding: 2px 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer"""
    try:
        # Load from the local directory
        model_dir = "./t5_summarization_model"
        
        # Check if model files exist
        if not os.path.exists(model_dir):
            st.error(f"Model directory '{model_dir}' not found!")
            return None, None, None
            
        st.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def generate_summary(model, tokenizer, device, text, max_length=128, num_beams=4):
    """Generate summary using the fine-tuned model"""
    try:
        # Preprocess text - add T5 prefix
        source_prefix = "summarize: "
        input_text = source_prefix + text
        
        # Tokenize
        inputs = tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.8
            )
        
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def calculate_text_stats(original_text, summary_text):
    """Calculate text statistics"""
    words_original = len(original_text.split())
    words_summary = len(summary_text.split())
    sentences_original = len(sent_tokenize(original_text))
    sentences_summary = len(sent_tokenize(summary_text))
    
    if words_original > 0:
        compression_ratio = (1 - words_summary / words_original) * 100
    else:
        compression_ratio = 0
    
    return {
        'original_words': words_original,
        'summary_words': words_summary,
        'original_sentences': sentences_original,
        'summary_sentences': sentences_summary,
        'compression_ratio': compression_ratio
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Transform long articles into concise summaries using fine-tuned T5 model")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("### Summary Configuration")
    
    max_length = st.sidebar.slider(
        "Maximum Summary Length (words)",
        min_value=50,
        max_value=200,
        value=128,
        help="Maximum number of words in the summary"
    )
    
    num_beams = st.sidebar.slider(
        "Number of Beams",
        min_value=1,
        max_value=8,
        value=4,
        help="Higher values produce better quality but take longer"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Model Information")
    st.sidebar.info("""
    **Model:** Fine-tuned T5-base  
    **Training:** CNN/DailyMail dataset  
    **Task:** Text Summarization
    """)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("""
        ❌ Model failed to load. Please ensure:
        - All model files are in 't5_summarization_model' folder
        - Files include: config.json, model.safetensors, tokenizer files
        - The folder is in the same directory as app.py
        """)
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Sample text"],
            horizontal=True
        )
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter your text to summarize:",
                height=300,
                placeholder="Paste your article, document, or any long text here...\n\nMinimum recommended length: 200 characters for good results.",
                help="The model works best with well-structured articles and documents"
            )
        else:
            sample_texts = {
                "Technology News": """
                Apple Inc. has unveiled its latest smartphone lineup, the iPhone 15 series, marking a significant upgrade from previous generations. The new devices feature titanium frames, improved camera systems with 48-megapixel main sensors, and the powerful A17 Pro chip. Notably, this generation transitions from Apple's proprietary Lightning connector to USB-C, aligning with European Union regulations and consumer demands for universal charging standards.

                The Pro models introduce customizable Action buttons replacing the mute switch, while all models now feature Dynamic Island as standard. Battery life has been enhanced across the series, with the iPhone 15 Plus offering up to 26 hours of video playback. The company also emphasized its environmental commitments, using 100% recycled cobalt in batteries and 75% recycled aluminum in enclosures.

                Pre-orders begin Friday with shipping starting the following week. Analysts project strong initial demand, particularly for the Pro models featuring the new tetraprism telephoto lens capable of 5x optical zoom. The pricing remains consistent with previous generations, starting at $799 for the base model.
                """,
                
                "Climate Change Report": """
                A comprehensive United Nations report released today indicates that climate change impacts are accelerating at an unprecedented rate, exceeding previous scientific projections. The study, compiled by the Intergovernmental Panel on Climate Change (IPCC), reveals that global temperatures have already risen 1.2°C above pre-industrial levels, approaching the critical 1.5°C threshold established in the Paris Agreement.

                The document highlights increasingly frequent and intense extreme weather events, including wildfires in North America, floods in Asia, and hurricanes in the Atlantic. Ocean temperatures have reached record highs, contributing to coral bleaching and sea-level rise threatening coastal communities. The report emphasizes that current national commitments to reduce greenhouse gas emissions remain insufficient to prevent catastrophic warming scenarios.

                Scientists urge immediate, drastic action including rapid transition to renewable energy, enhanced carbon capture technologies, and international cooperation. The findings will inform discussions at the upcoming COP28 climate summit, where world leaders are expected to strengthen climate commitments and establish new funding mechanisms for developing nations.
                """,
                
                "Scientific Discovery": """
                NASA's Perseverance rover has achieved a major milestone in Martian exploration, successfully collecting and caching the first rock core samples from Jezero Crater that will eventually be returned to Earth. The samples, extracted from an ancient river delta environment, show promising signs of preserving evidence of past microbial life. Geological analysis indicates the rocks formed in a watery environment rich in minerals that could have supported living organisms.

                The rover's sophisticated instruments have detected organic molecules within the samples, though scientists caution this doesn't conclusively prove past life existence. The collection represents the beginning of the Mars Sample Return campaign, a joint NASA-ESA endeavor planning to retrieve these samples in the early 2030s. Perseverance has collected 20 samples so far, each sealed in ultra-clean tubes to prevent contamination.

                Mission controllers have extended the rover's operations to explore new areas of scientific interest within the crater. The findings from these samples could fundamentally reshape our understanding of Mars' history and the potential for life elsewhere in our solar system.
                """
            }
            
            selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
            input_text = st.text_area(
                "Sample text:",
                value=sample_texts[selected_sample],
                height=300,
                help="Feel free to modify the sample text or use your own"
            )
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 50:
                st.warning("⚠️ Please enter at least 50 characters of text to summarize.")
            else:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                status_text.text("Generating summary...")
                
                # Generate summary
                summary = generate_summary(
                    model, tokenizer, device, 
                    input_text, max_length, num_beams
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if summary:
                    # Display summary
                    st.markdown("### ✅ Summary Result")
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate and display statistics
                    stats = calculate_text_stats(input_text, summary)
                    
                    st.markdown("### 📊 Text Statistics")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric(
                            "Original Words", 
                            stats['original_words'],
                            help="Total words in original text"
                        )
                    
                    with col_stat2:
                        st.metric(
                            "Summary Words", 
                            stats['summary_words'],
                            help="Total words in generated summary"
                        )
                    
                    with col_stat3:
                        st.metric(
                            "Compression Ratio", 
                            f"{stats['compression_ratio']:.1f}%",
                            help="Percentage of text reduced"
                        )
                    
                    with col_stat4:
                        st.metric(
                            "Reading Time Saved", 
                            f"{(stats['original_words'] - stats['summary_words']) // 200} min",
                            help="Estimated reading time saved (at 200 wpm)"
                        )
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name="ai_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate summary. Please try again with different text or settings.")
        
        # Instructions when no summary generated yet
        else:
            st.info("""
            **Ready to summarize!** 
            
            - Enter your text in the left panel
            - Adjust settings in the sidebar if needed  
            - Click the **'Generate Summary'** button above
            - Your concise summary will appear here
            """)
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        **📝 Text Quality**
        - Use well-structured articles
        - Ensure proper grammar and spelling
        - Avoid excessive formatting
        """)
    
    with tip_col2:
        st.markdown("""
        **⚙️ Settings Guide**
        - Increase beam count for better quality
        - Adjust length based on content complexity
        - Use 4-6 beams for optimal results
        """)
    
    with tip_col3:
        st.markdown("""
        **🎯 Best Use Cases**
        - News articles
        - Research papers
        - Long documents
        - Blog posts
        - Reports
        """)

if __name__ == "__main__":
    main()