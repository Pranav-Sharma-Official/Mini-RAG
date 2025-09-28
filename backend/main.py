import os
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Initialize Clients (Models will be loaded lazily) ---

# Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
# NOTE: Use a model name you have verified is available with your API key
llm = genai.GenerativeModel('gemini-pro')

# Define collection name for Qdrant
COLLECTION_NAME = "my_rag_collection"

# --- Lazy Loading for ML Models ---
# Global variables to hold the models. They are initialized to None.
embedding_model = None
reranker_model = None

def get_embedding_model():
    """Loads the embedding model if it's not already loaded."""
    global embedding_model
    if embedding_model is None:
        print("Initializing embedding model for the first time...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("Embedding model initialized.")
    return embedding_model

def get_reranker_model():
    """Loads the reranker model if it's not already loaded."""
    global reranker_model
    if reranker_model is None:
        print("Initializing reranker model for the first time...")
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        print("Reranker model initialized.")
    return reranker_model

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def setup_qdrant_collection():
    """Creates the Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        # Create the collection with the hardcoded vector size
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  # FIX: Hardcoded dimension for 'all-MiniLM-L6-v2' to prevent OOM errors
                distance=models.Distance.COSINE
            )
        )

@app.on_event("startup")
def startup_event():
    """This runs when the app starts up. It will now NOT load any models."""
    setup_qdrant_collection()

# --- API Endpoints ---
@app.post("/api/upload-and-process/")
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
        model = get_embedding_model() # Model is loaded here, on first API call
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk.page_content).tolist()
            points.append(models.PointStruct(
                id=i, 
                vector=embedding, 
                payload={"text": chunk.page_content}
            ))
        
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, # FIX: Use hardcoded dimension here as well
                distance=models.Distance.COSINE
            )
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return {"message": f"Successfully processed and indexed '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)


@app.post("/api/query/")
async def query_rag(query: str = Form(...)):
    try:
        query_embedding = get_embedding_model().encode(query).tolist() # Model loaded on first call

        retrieved_hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=10
        )
        retrieved_docs = [hit.payload['text'] for hit in retrieved_hits]

        reranker = get_reranker_model() # Reranker model loaded on first call
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
