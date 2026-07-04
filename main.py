import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from agent import web_search_answer
from rag import get_answer
from spectrum import full_spectrum_answer

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INDEX_PATH = "./faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class AskRequest(BaseModel):
    question: str
    mode: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/index-status")
def index_status():
    return {"indexed": os.path.isdir(INDEX_PATH)}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        FAISS.from_documents(chunks, embeddings).save_local(INDEX_PATH)
        return {"status": "ok", "chunks": len(chunks)}
    finally:
        os.unlink(tmp_path)


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode == 1:
        return get_answer(req.question)

    if req.mode == 2:
        doc_result = get_answer(req.question)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        prompt = (
            f"Answer the question using the document context below. "
            f"If the document context is insufficient, you may supplement from your own knowledge "
            f"but explicitly prefix any addition with 'From general knowledge: '. "
            f"Cite document page sources for document-based claims.\n\n"
            f"Context: {doc_result['answer']}\n\nQuestion: {req.question}"
        )
        answer = llm.invoke(prompt).content
        return {"answer": answer, "sources": doc_result["sources"]}

    if req.mode == 3:
        return web_search_answer(req.question)

    if req.mode == 4:
        return full_spectrum_answer(req.question)

    raise HTTPException(status_code=400, detail="mode must be 1, 2, 3, or 4")
