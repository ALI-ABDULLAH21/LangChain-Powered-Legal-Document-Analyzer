import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the legal text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store legal text embeddings into FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get a conversational chain tailored for legal Q&A
def get_conversational_chain():
    # Modified prompt template for legal context
    prompt_template = """
    You are an AI specialized in legal analysis. Answer the question in a legally precise manner, based on the provided legal context.
    If the answer is not found in the context, respond with "The answer is not available in the provided legal context."
    Do not provide speculative or incorrect answers.\n\n
    Legal Context:\n{context}\n
    Legal Question:\n{question}\n
    Legal Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to process user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Legal Answer: ", response["output_text"])

# Main function to drive the Streamlit app
def main():
    st.set_page_config(page_title="Legal Document Analysis Chat")
    st.header("Chat with Legal Documents and get AI-based Legal Assistance")

    # User input for legal question
    user_question = st.text_input("Ask a Legal Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar to upload PDF documents
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Legal Documents (PDF) and click Submit to Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Legal documents processed successfully!")

if __name__ == "__main__":
    main()
