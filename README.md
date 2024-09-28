

# Legal Document Analysis Chatbot

This project is an AI-powered legal document analysis tool that allows users to upload PDF files containing legal documents, ask legal questions related to the content, and receive detailed, accurate responses from an AI model trained to work with legal contexts. The chatbot leverages Google Gemini Pro and LangChain for efficient natural language processing and question-answering tasks.

## Features

- **Legal Document Upload**: Upload one or multiple legal PDF documents.
- **Text Extraction**: Automatically extracts text from the uploaded PDFs for further analysis.
- **Text Chunking**: Splits the legal text into manageable chunks to ensure effective analysis.
- **AI-Powered Question-Answering**: Ask questions about the legal content, and the AI will provide accurate responses based on the uploaded documents.
- **Legal Precision**: The chatbot provides answers only from the given legal context, avoiding speculative or incorrect responses.
- **FAISS Vector Search**: Efficiently searches the most relevant sections of the legal documents using FAISS (Facebook AI Similarity Search).

## Technology Stack

- **Streamlit**: For building the interactive web-based UI.
- **PyPDF2**: For extracting text from PDF documents.
- **LangChain**: For managing the natural language processing pipeline and handling document-based question-answering.
- **Google Gemini Pro**: AI model used for embeddings and generating responses.
- **FAISS**: Vector store for document similarity search.
- **Python**: Core programming language for development.

## Prerequisites

- Python 3.8 or above
- A Google API Key (for Google Generative AI SDK)
- Install the required Python packages (listed below)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ALI-ABDULLAH21/legal-document-analysis-chatbot.git
cd legal-document-analysis-chatbot
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install all required libraries using `pip`.

```bash
pip install -r requirements.txt
```

Make sure your `requirements.txt` file includes these dependencies:
```txt
streamlit
PyPDF2
langchain
faiss-cpu
python-dotenv
google-generativeai
```

### 4. Set Up Environment Variables

- Create a `.env` file in the root directory of your project.
- Add your **Google API Key** in the `.env` file.

Example `.env` file:
```env
GOOGLE_API_KEY=your-google-api-key
```

### 5. Run the Application

Once everything is set up, you can run the Streamlit app using the following command:

```bash
streamlit run app.py
```

### 6. Upload Legal PDFs and Start Asking Questions

- Navigate to the sidebar to upload one or more legal documents in PDF format.
- Once processed, you can ask questions in the input field related to the legal text in the documents.

## Example Usage

1. **Upload Legal Documents**: Upload contracts, laws, or court rulings as PDF files.
2. **Ask Questions**: Enter a question like:
   - "What are the obligations of the parties under this contract?"
   - "What is the penalty for breach of contract as per this agreement?"
   - "What legal rights are provided under this law?"
3. **Receive AI-Generated Answers**: The AI will return detailed answers based on the legal documents you've uploaded.

## Limitations

- The tool relies on the accuracy and completeness of the uploaded legal documents.
- The AI model provides answers based only on the content of the provided documents and will not perform general legal research or offer legal advice.
- Large legal documents may take time to process depending on their size.

## Future Improvements

- **Multi-language Support**: Extend the tool to handle legal documents in different languages.
- **Enhanced Document Management**: Add support for storing and managing documents in a database.
- **Model Fine-Tuning**: Fine-tune models for specific legal domains (e.g., criminal law, contract law, property law).



