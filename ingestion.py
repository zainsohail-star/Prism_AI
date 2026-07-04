import argparse
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_PATH = "./faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main() -> None:
    # --- CLI ---
    parser = argparse.ArgumentParser(description="Ingest a PDF into a FAISS vector store.")
    parser.add_argument("--file", required=True, help="Path to the PDF file to ingest.")
    args = parser.parse_args()

    pdf_path: str = args.file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Load ---
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # --- Chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # --- Embed & index ---
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)

    print(
        f"Ingested {len(chunks)} chunks from {os.path.basename(pdf_path)}"
        f" → saved to {INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
