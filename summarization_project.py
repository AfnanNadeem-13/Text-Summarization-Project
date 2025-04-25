## Import required libraries
import pandas as pd
import spacy
from transformers import pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# Load dataset
def load_data():
    # Load CSV files
    train_data = pd.read_csv('train.csv')
    validation_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    
    # Preview data
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
        # Truncate text to avoid long input errors
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
