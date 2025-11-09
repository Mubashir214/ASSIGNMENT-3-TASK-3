import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time

# Set page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
        line-height: 1.6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer from root directory"""
    try:
        # Since all files are in root, we load from current directory "."
        st.info("🔍 Loading model from current directory...")
        
        # Check if essential model files exist in root
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ Missing files: {', '.join(missing_files)}")
            st.info("Current directory files:")
            st.write(os.listdir('.'))
            return None, None, None
        
        st.success("✅ All model files found! Loading tokenizer and model...")
        
        # Load from current directory
        tokenizer = AutoTokenizer.from_pretrained(".")
        model = AutoModelForSeq2SeqLM.from_pretrained(".")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        # Show what files are available for debugging
        st.info("📁 Files in current directory:")
        st.write(os.listdir('.'))
        return None, None, None

def generate_summary(model, tokenizer, device, text, max_length=100, num_beams=2):
    """Generate summary using the fine-tuned model"""
    try:
        # Add T5 prefix
        input_text = "summarize: " + text
        
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
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Transform long articles into concise summaries")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    max_length = st.sidebar.slider("Summary Length", 50, 150, 100)
    num_beams = st.sidebar.slider("Quality", 1, 4, 2)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("""
        ❌ Model failed to load. 
        
        **Please ensure all these files are in your root directory:**
        - config.json
        - generation_config.json  
        - model.safetensors
        - special_tokens_map.json
        - spiece.model
        - tokenizer.json
        - tokenizer_config.json
        - training_args.bin
        
        **Current files detected:** (see above)
        """)
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        sample_texts = {
            "Technology News": """
            Apple Inc. has unveiled its latest smartphone lineup, the iPhone 15 series, marking a significant upgrade from previous generations. 
            The new devices feature titanium frames, improved camera systems with 48-megapixel main sensors, and the powerful A17 Pro chip. 
            Notably, this generation transitions from Apple's proprietary Lightning connector to USB-C, aligning with European Union regulations.
            """,
            "Climate Change": """
            A comprehensive United Nations report indicates that climate change impacts are accelerating at an unprecedented rate. 
            The study reveals that global temperatures have already risen 1.2°C above pre-industrial levels, approaching the critical 1.5°C threshold.
            Scientists urge immediate action to reduce greenhouse gas emissions and transition to renewable energy sources.
            """,
            "Space Exploration": """
            NASA's Perseverance rover has successfully collected the first rock core samples from Mars that will be returned to Earth. 
            The samples, extracted from an ancient river delta environment, show promising signs of preserving evidence of past microbial life.
            This represents a major milestone in the search for extraterrestrial life within our solar system.
            """
        }
        
        input_method = st.radio("Choose input method:", ["Type text", "Use sample"], horizontal=True)
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter your text to summarize:",
                height=250,
                placeholder="Paste your article, document, or any long text here...",
                help="The model works best with well-structured articles of 200+ words"
            )
        else:
            selected_sample = st.selectbox("Choose sample text:", list(sample_texts.keys()))
            input_text = st.text_area(
                "Sample text:",
                value=sample_texts[selected_sample],
                height=250
            )
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 30:
                st.warning("⚠️ Please enter at least 30 characters of text.")
            else:
                with st.spinner("🔄 Generating summary..."):
                    summary = generate_summary(model, tokenizer, device, input_text, max_length, num_beams)
                
                if summary:
                    st.markdown("### ✅ Summary Result")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Calculate statistics
                    orig_words = len(input_text.split())
                    sum_words = len(summary.split())
                    compression = ((orig_words - sum_words) / orig_words) * 100 if orig_words > 0 else 0
                    
                    # Display stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", orig_words)
                    with col2:
                        st.metric("Summary Words", sum_words)
                    with col3:
                        st.metric("Reduction", f"{compression:.1f}%")
                    
                    # Download button
                    st.download_button(
                        "💾 Download Summary",
                        summary,
                        file_name="ai_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate summary. Please try again with different text.")
        else:
            st.info("""
            **Ready to summarize!**
            
            • Enter your text in the left panel
            • Click the **Generate Summary** button
            • Get your concise summary here
            """)

if __name__ == "__main__":
    main()
