from transformers import pipeline, AutoTokenizer
import fitz
import streamlit as st



st.title("Summarizer")
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, tokenizer, max_len=999):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


# Streamlit file uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.info("Extracting text from PDF...")
    output_text = extract_text_from_pdf("temp_uploaded.pdf")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", local_files_only=True)
    tokenizer.model_max_length = 1024
    max_input_length = tokenizer.model_max_length
    summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=0)

    if len(tokenizer.encode(output_text,add_special_tokens=False)) > max_input_length:
        st.warning("The text is too long, chunking...")
        chunks = chunk_text(output_text, tokenizer)
        st.info("Text has been chunked into smaller parts for summarization.")
    else:
        chunks = [output_text]

    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {i + 1}/{len(chunks)}")
        is_last_chunk = i == len(chunks) - 1

        # Safely re-tokenize for length
        encoded_chunk = tokenizer.encode(chunk, add_special_tokens=False)
        token_length = len(encoded_chunk)

        # Log length for debugging
        print(f"Chunk {i + 1} has {token_length} tokens")

        if not chunk.strip():
            st.warning(f"Chunk {i + 1} is empty, skipping.")
            summaries.append("")
            progress_bar.progress((i + 1) / len(chunks))
            continue

        if token_length == 0:
            st.warning(f"Chunk {i + 1} tokenized to 0 tokens, skipping.")
            summaries.append("")
            progress_bar.progress((i + 1) / len(chunks))
            continue

        if token_length < 10 and not is_last_chunk:
            st.warning(f"Chunk {i + 1} too short (<10 tokens), skipping.")
            summaries.append("")
            progress_bar.progress((i + 1) / len(chunks))
            continue

        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False,truncation=True)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error summarizing chunk {i + 1}: {e}")
            summaries.append("")
        progress_bar.progress((i + 1) / len(chunks))

    final_summary = "\n\n".join(summaries)

    st.subheader("Summary:")
    st.write(final_summary)
