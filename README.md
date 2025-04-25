# Text Summarization Project

## Objective:
Create a system that summarizes lengthy articles, blogs, or news into concise summaries.

## Dataset:
- **CNN/Daily Mail Dataset**
  - The dataset contains news articles and their corresponding summaries.
  - The dataset files used in this project include:
    - `train.csv` – Training data
    - `validation.csv` – Validation data
    - `test.csv` – Test data

## Technologies Used:
- **Python 3.x**
- **Libraries**:
  - `pandas`: For data manipulation and reading CSV files.
  - `spaCy`: For extractive summarization.
  - `transformers` (HuggingFace): For abstractive summarization using pre-trained models like T5.
  - `torch`: PyTorch, required for running HuggingFace models.
  - `tqdm`: For displaying a progress bar during summarization.
- **Pre-Trained Models**:
  - **T5 (Text-to-Text Transfer Transformer)** from HuggingFace for abstractive summarization.
  - **spaCy**'s `en_core_web_sm` model for extractive summarization.

## Steps:

### 1. Preprocessing the Text:
   - The text data is preprocessed by converting the text to lowercase and stripping extra spaces.

### 2. Extractive Summarization using spaCy:
   - Extractive summarization is performed by selecting the first three sentences from the article. This is done using the spaCy model (`en_core_web_sm`).

### 3. Abstractive Summarization using HuggingFace:
   - The T5 model (`t5-small`) from HuggingFace is used to generate an abstractive summary of the article. 
   - The article text is truncated to 512 characters to avoid input length errors with the model.

### 4. Fine-tuning Models (Optional):
   - Fine-tuning could be performed if needed, but for simplicity, the pre-trained T5 model is used as-is in this project.

### 5. Evaluating Summary Coherence:
   - The generated summaries are saved into CSV files for each dataset (training, validation, and test).
   - Each row contains both an extractive and an abstractive summary.

## Code Description:

### Main Python Script:
```python
# Import required libraries
import pandas as pd
import spacy
from transformers import pipeline
from tqdm import tqdm
import time

# Load dataset
def load_data():
    train_data = pd.read_csv('train.csv')
    validation_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    print("Training Data Sample:\n", train_data.head())
    print("Validation Data Sample:\n", validation_data.head())
    print("Test Data Sample:\n", test_data.head())
    return train_data, validation_data, test_data

# Preprocess the text
def preprocess_text(text):
    return text.lower().strip()

# Extractive Summarization using SpaCy
def extractive_summary(text, nlp):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return ' '.join(sentences[:3])  # Top 3 sentences

# Abstractive Summarization using HuggingFace
def abstractive_summary(text, summarizer):
    try:
        summary = summarizer(text[:512], max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Error generating summary"

# Summarize the entire dataset (with optional row limit for speed)
def summarize_data(data, nlp, summarizer, limit=None):
    summaries = []
    if limit:
        data = data.head(limit)
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Summarizing Articles"):
        article = preprocess_text(row['article'])
        extractive_result = extractive_summary(article, nlp)
        abstractive_result = abstractive_summary(article, summarizer)
        summaries.append([extractive_result, abstractive_result])
    return pd.DataFrame(summaries, columns=['Extractive Summary', 'Abstractive Summary'])

# Save summaries to a CSV file
def save_summaries(summary_df, filename="summarized_articles.csv"):
    summary_df.to_csv(filename, index=False)
    print(f"Summaries saved to {filename}")

# Main function
def main():
    print("Text Summarization Project Started 🚀")
    
    # Load models
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    print("Loading HuggingFace summarization model (t5-small)...")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    # Load datasets
    print("Loading dataset...")
    train_data, validation_data, test_data = load_data()

    # Summarize training data
    print("\nProcessing Training Data...")
    start_time = time.time()
    train_summaries = summarize_data(train_data, nlp, summarizer, limit=10)
    print(f"Training Data Summarization Completed in {time.time() - start_time:.2f} seconds")
    save_summaries(train_summaries, "train_summaries.csv")

    # Summarize validation data
    print("\nProcessing Validation Data...")
    start_time = time.time()
    validation_summaries = summarize_data(validation_data, nlp, summarizer, limit=10)
    print(f"Validation Data Summarization Completed in {time.time() - start_time:.2f} seconds")
    save_summaries(validation_summaries, "validation_summaries.csv")

    # Summarize test data
    print("\nProcessing Test Data...")
    start_time = time.time()
    test_summaries = summarize_data(test_data, nlp, summarizer, limit=10)
    print(f"Test Data Summarization Completed in {time.time() - start_time:.2f} seconds")
    save_summaries(test_summaries, "test_summaries.csv")

if __name__ == "__main__":
    main()
