import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Set page config
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

def initialize_model():
    """Initialize model only when needed"""
    try:
        st.info("🚀 Initializing YOUR fine-tuned model... Please wait.")
        
        # Load tokenizer first (lightweight)
        tokenizer = AutoTokenizer.from_pretrained(".")
        st.session_state.tokenizer = tokenizer
        
        # Load model with optimizations for large models
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ".",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="./offload"
        )
        
        st.session_state.model = model
        st.session_state.model_initialized = True
        st.success("✅ YOUR fine-tuned model loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Model initialization failed: {str(e)}")
        return False
    return True

def generate_summary_lazy(text, max_length=100):
    """Generate summary with lazy-loaded model"""
    try:
        if not st.session_state.model_initialized:
            if not initialize_model():
                return None
        
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        
        # Prepare input
        input_text = "summarize: " + text
        inputs = tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Using YOUR Fine-tuned Model")
    
    # File check
    with st.sidebar:
        st.subheader("📁 Model Files Status")
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        all_files_ok = True
        
        for file in required_files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
                all_files_ok = False
        
        if not all_files_ok:
            st.error("Missing model files! Upload all files to root directory.")
            return
    
    # Settings
    with st.sidebar:
        st.subheader("⚙️ Settings")
        max_length = st.slider("Summary Length", 50, 150, 100)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        sample_texts = {
            "Technology": """Apple's new iPhone features advanced cameras and faster processors.""",
            "Science": """NASA's rover discovered evidence of ancient water on Mars.""",
            "News": """The company announced record profits this quarter."""
        }
        
        input_method = st.radio("Input:", ["Type text", "Quick sample"], horizontal=True)
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter text:",
                height=250,
                placeholder="Paste your article here..."
            )
        else:
            selected = st.selectbox("Sample:", list(sample_texts.keys()))
            input_text = sample_texts[selected]
            st.text_area("Sample text:", value=input_text, height=150)
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 10:
                st.warning("Please enter some text.")
            else:
                with st.spinner("🔄 Generating summary with YOUR model..."):
                    summary = generate_summary_lazy(input_text, max_length)
                
                if summary:
                    st.markdown("### ✅ Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Words", len(input_text.split()))
                    with col2:
                        st.metric("Summary Words", len(summary.split()))
                    
                    # Download
                    st.download_button(
                        "💾 Download Summary",
                        summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate summary. The model might be still loading.")

if __name__ == "__main__":
    main()
