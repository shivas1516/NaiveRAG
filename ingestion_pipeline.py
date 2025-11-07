from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def ingest_pdf(file_path: str):

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the PDF document and extract the text
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # chunk the text into smaller pieces, with default separator in RecursiveCharacterTextSplitter, chunk size of 1000 and overlap of 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(docs)

    # Using embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create In-Memory Vector Store and upload  the embedded text chunks
    vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

    return vector_store, split_docs