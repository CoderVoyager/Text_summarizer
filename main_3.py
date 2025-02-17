import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

# Model and tokenizer loading
checkpoint = "MBZUAI/LaMini-Flan-T5-248M" # You can try larger models like "google/flan-t5-large" or "google/flan-t5-xl" if resources allow
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

# File loader and preprocessing
def file_preprocessing(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        # Adjusted chunk size and overlap - Experiment with these values
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100) # Increased chunk size, overlap
        texts = text_splitter.split_documents(pages)

        final_texts = " ".join([text.page_content for text in texts])
        return final_texts
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# LLM pipeline
def llm_pipeline(file_path):
    input_text = file_preprocessing(file_path)
    if not input_text:
        return "Error: Unable to process the file."

    summarizer = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=1000,  # Adjusted max length to be more reasonable
        min_length=50,
        do_sample=True, # Added sampling for more diverse summaries
        temperature=0.8, # Adjusted temperature for controlled randomness
        top_p=0.9     # Adjusted top_p for nucleus sampling
    )

    result = summarizer(input_text)
    return result[0]['summary_text']

@st.cache_data
# Function to display the PDF
def displayPDF(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Streamlit app setup
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        file_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)  # Ensure 'data' directory exists

        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded File")
                displayPDF(file_path)

            with col2:
                summary = llm_pipeline(file_path)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()