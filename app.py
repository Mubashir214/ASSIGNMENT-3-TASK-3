import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import os
import time

# Download NLTK data quietly
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

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
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned model and tokenizer"""
    try:
        # Load from the local directory
        model_dir = "./t5_summarization_model"
        
        # Check if model files exist
        if not os.path.exists(model_dir):
            return None, None, None, "Model directory not found"
            
        # Check for essential files
        essential_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        for file in essential_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                return None, None, None, f"Missing essential file: {file}"
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device, "Success"
        
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

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
            padding=True,
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
                no_repeat_ngram_size=2
            )
        
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def calculate_text_stats(original_text, summary_text):
    """Calculate text statistics"""
    try:
        words_original = len(original_text.split())
        words_summary = len(summary_text.split())
        
        if words_original > 0:
            compression_ratio = (1 - words_summary / words_original) * 100
        else:
            compression_ratio = 0
        
        return {
            'original_words': words_original,
            'summary_words': words_summary,
            'compression_ratio': compression_ratio
        }
    except:
        return {
            'original_words': 0,
            'summary_words': 0,
            'compression_ratio': 0
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Transform long articles into concise summaries")
    
    # Initialize session state
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = ""
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    max_length = st.sidebar.slider(
        "Summary Length",
        min_value=50,
        max_value=150,
        value=100,
        help="Maximum words in summary"
    )
    
    num_beams = st.sidebar.slider(
        "Quality Setting",
        min_value=1,
        max_value=4,
        value=2,
        help="Higher = better quality, slower generation"
    )
    
    # Load model with spinner
    with st.sidebar:
        with st.spinner("Loading AI model..."):
            model, tokenizer, device, load_status = load_model()
    
    if model is None:
        st.error(f"❌ Model loading failed: {load_status}")
        st.info("""
        **Troubleshooting tips:**
        - Ensure all model files are uploaded
        - Check that 't5_summarization_model' folder exists
        - Verify files include: config.json, model.safetensors, tokenizer files
        - Try reloading the app
        """)
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        sample_texts = {
            "Technology": """
            Apple has announced the launch of its new iPhone 15 series, featuring improved cameras, 
            a faster A17 processor, and longer battery life. The new models include USB-C charging, 
            a first for iPhones. Pre-orders begin this week with shipping expected next month. 
            Analysts predict strong sales due to high consumer demand.
            """,
            "Environment": """
            The United Nations has issued a warning about climate change effects worldwide. 
            Rising temperatures are causing more frequent wildfires, floods, and hurricanes. 
            Governments are urged to take urgent action to reduce carbon emissions.
            """,
            "Science": """
            NASA's Perseverance rover has collected samples of Martian rock. The samples will be 
            returned to Earth by a future mission. The rover has been exploring Jezero Crater 
            since its landing in 2021.
            """
        }
        
        input_method = st.radio("Input method:", ["Type text", "Use sample"], horizontal=True)
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter text to summarize:",
                height=250,
                placeholder="Paste your article or text here...",
                key="input_text"
            )
        else:
            selected_sample = st.selectbox("Choose sample:", list(sample_texts.keys()))
            input_text = st.text_area(
                "Sample text:",
                value=sample_texts[selected_sample],
                height=250,
                key="sample_text"
            )
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        generate_clicked = st.button(
            "🚀 Generate Summary", 
            type="primary", 
            use_container_width=True,
            disabled=(model is None)
        )
        
        if generate_clicked and input_text.strip():
            if len(input_text.strip()) < 30:
                st.warning("Please enter at least 30 characters.")
            else:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                status_text.text("Generating summary...")
                
                # Generate summary
                summary = generate_summary(model, tokenizer, device, input_text, max_length, num_beams)
                
                progress_bar.empty()
                status_text.empty()
                
                if summary:
                    st.session_state.summary_generated = True
                    st.session_state.current_summary = summary
                    
                    # Display summary
                    st.markdown("### ✅ Summary")
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    stats = calculate_text_stats(input_text, summary)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", stats['original_words'])
                    with col2:
                        st.metric("Summary Words", stats['summary_words'])
                    with col3:
                        st.metric("Reduction", f"{stats['compression_ratio']:.1f}%")
                    
                    # Download
                    st.download_button(
                        "💾 Download Summary",
                        summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to generate summary. Please try again.")
        
        elif st.session_state.summary_generated:
            st.markdown("### ✅ Summary")
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.write(st.session_state.current_summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button(
                "💾 Download Summary",
                st.session_state.current_summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("""
            **Ready to summarize!**
            
            • Enter text in the left panel
            • Click 'Generate Summary' 
            • Get your concise summary here
            """)

if __name__ == "__main__":
    main()
