🚀 Task 1 — BERT: Customer Feedback Sentiment Classification
Problem

Classify customer feedback into Positive, Negative, or Neutral sentiments.

Dataset

Kaggle: Customer Feedback Sentiment Dataset
(https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset
)

Deliverables

✔ Preprocessing + Tokenization
✔ Training + Validation Pipeline
✔ Evaluation Metrics
✔ Example Predictions

Evaluation Metrics

Accuracy

Precision / Recall / F1-Score

Confusion Matrix

🤖 Task 2 — GPT-2/LLaMA: Pseudo-code → Python Code Generation
Problem

Translate structured pseudo-code into syntactically and semantically correct Python code.

Dataset

SPOC (Student Pseudo-code to Code) Dataset
https://github.com/sumith1896/spoc

Research Paper

https://arxiv.org/pdf/1906.04908

Deliverables

✔ Preprocessing of pseudo-code/code pairs
✔ Tokenization + Formatting
✔ Fine-tuning GPT-2 (Causal LM)
✔ Evaluation Metrics:
– BLEU
– CodeBLEU
– Human Evaluation
✔ Streamlit / Gradio Interface

Output Example

User enters pseudo-code → Model returns working Python code.

📝 Task 3 — T5/BART: Abstractive Text Summarization
Problem

Generate concise summaries from long news articles.

Dataset

Kaggle (CNN-DailyMail Summarization Dataset)
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Deliverables

✔ Dataset Preprocessing
✔ Model Fine-Tuning (T5/BART)
✔ ROUGE Evaluation
✔ Example Summaries

Evaluation Metrics

ROUGE-1

ROUGE-2

ROUGE-L

Human evaluation for readability + relevance

🛠 Installation
git clone <repo-url>
cd project-folder
pip install -r requirements.txt

▶️ How to Run Each Task
Task 1 (BERT)
cd Task1_BERT_Sentiment
python preprocess.py
python train.py
python evaluate.py

Task 2 (GPT-2/LLaMA)
cd Task2_GPT_Pseudocode2Code
python preprocess_pairs.py
python finetune_gpt2.py
python evaluate_metrics.py
streamlit run app.py

Task 3 (T5/BART)
cd Task3_T5_Summarization
python preprocess.py
python finetune_t5.py
python evaluate_summarizer.py

📊 Results

Each task folder contains:

Metrics report

Model checkpoints

Example outputs

Plots (accuracy, loss curves, ROUGE, BLEU, etc.)

🙌 Acknowledgements

BERT, GPT-2, T5, BART — Hugging Face Transformers

Kaggle + SPOC Dataset creators

Research papers referenced in each task
