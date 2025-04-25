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
- **Python **
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

