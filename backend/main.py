import os
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
llm = genai.GenerativeModel('gemini-2.5-pro')
COLLECTION_NAME = "my_rag_collection"

embedding_model = None
reranker_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return embedding_model

def get_reranker_model():
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return reranker_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'static' directory to serve files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def startup_event():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

# NEW: Root endpoint to serve the index.html file
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/upload-and-process/")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        points = []
        model = get_embedding_model()
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk.page_content).tolist()
            points.append(models.PointStruct(id=i, vector=embedding, payload={"text": chunk.page_content}))
        
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return {"message": f"Successfully processed and indexed '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)

@app.post("/query/")
async def query_rag(query: str = Form(...)):
    try:
        query_embedding = get_embedding_model().encode(query).tolist()
        retrieved_hits = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=10)
        retrieved_docs = [hit.payload['text'] for hit in retrieved_hits]
        reranker = get_reranker_model()
        rerank_pairs = [[query, doc] for doc in retrieved_docs]
        rerank_scores = reranker.predict(rerank_pairs)
        reranked_docs_with_scores = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        final_context_docs = [doc for doc, score in reranked_docs_with_scores[:3]]
        context_str = "\n\n".join(final_context_docs)
        prompt = f"""
        You are a helpful AI assistant. Answer the user's question based *only* on the following context.
        If the answer is not available in the context, say "I cannot answer this question based on the provided document."

        Context:
        {context_str}

        Question:
        {query}

        Answer:
        """
        response = llm.generate_content(prompt)
        return { "answer": response.text, "sources": final_context_docs }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {str(e)}")