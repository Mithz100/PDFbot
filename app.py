import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import pytesseract
from pdf2image import convert_from_bytes
import pickle

st.set_page_config(page_title="Document Questionaire", layout="wide")

st.markdown("""
## Document Questionaire: Get instant insights from your Documents

### How It Works

Follow these simple steps:

1. **Enter Your API Key**

2. **Upload Your Documents**

3. **Ask a Question**
""")

api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            st.write("Reading PDF file...")
            pdf_bytes = pdf.read()

            st.write("Extracting text from PDF using PyPDF2...")
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            st.write("Converting PDF to images for OCR...")
            images = convert_from_bytes(pdf_bytes)
            for image in images:
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text
            
            st.write("Text extraction and OCR completed.")
    except Exception as e:
        st.error(f"an error occured: {e}")
        st.stop()
        
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def check_faiss_index():
    if not os.path.exists("faiss_index/index.faiss"):
        raise FileNotFoundError("FAISS index file not found. Make sure to process your documents first.")

def user_input(user_question, api_key):
    check_faiss_index()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI clone chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
