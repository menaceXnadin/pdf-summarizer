# DocSummarizer

I built this just to get hands-on experience with Hugging Face models, Streamlit, and Transformers. It's a simple Streamlit app to upload PDF files and generate concise summaries using Hugging Face Transformers.

## Features
- Upload PDF files via a simple web interface
- Automatic extraction of text from PDFs
- Summarization using the BART model (`facebook/bart-large-cnn`)
- Handles long documents by chunking text
- Progress bar for summarization process

## Requirements
- Python 3.7+
- Streamlit
- transformers
- torch

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DocSummarizer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Open the provided local URL in your browser.
3. Upload a PDF file and view the generated summary.

## Notes
- The summarization model is loaded locally. Ensure you have downloaded the `facebook/bart-large-cnn` model or have internet access for the first run.
- GPU support is recommended for faster summarization.
