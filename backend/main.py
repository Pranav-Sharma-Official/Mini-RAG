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

# --- Initialize Models and Clients ---
# Initialize the embedding model (runs locally)
print("Initializing embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Embedding model initialized.")

# Initialize the reranker model (runs locally)
print("Initializing reranker model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
print("Reranker model initialized.")

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Configure Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
llm = genai.GenerativeModel('gemini-1.5-flash')

# Define collection name for Qdrant
COLLECTION_NAME = "my_rag_collection"

# --- FastAPI App ---
app = FastAPI()

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Helper Functions ---
def setup_qdrant_collection():
    """Creates the Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print("Collection created successfully.")

# Initialize the collection on startup
setup_qdrant_collection()

# --- API Endpoints ---
@app.post("/upload-and-process/")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    """Handles file upload, chunking, embedding, and storing in Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save the uploaded file temporarily
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load and split the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Embed and upsert chunks into Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk.page_content).tolist()
            points.append(models.PointStruct(
                id=i, 
                vector=embedding, 
                payload={"text": chunk.page_content}
            ))
        
        # Clear existing points before adding new ones
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        return {"message": f"Successfully processed and indexed '{file.filename}'."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        shutil.rmtree(temp_dir)


@app.post("/query/")
async def query_rag(query: str = Form(...)):
    """Handles user queries, retrieves, reranks, and generates an answer."""
    try:
        # 1. Embed the user's query
        query_embedding = embedding_model.encode(query).tolist()

        # 2. Retrieve relevant documents from Qdrant
        retrieved_hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=10  # Retrieve top 10 results initially
        )
        retrieved_docs = [hit.payload['text'] for hit in retrieved_hits]

        # 3. Rerank the retrieved documents
        rerank_pairs = [[query, doc] for doc in retrieved_docs]
        rerank_scores = reranker_model.predict(rerank_pairs)
        
        # Combine docs with their scores and sort
        reranked_docs_with_scores = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # Select top 3-5 documents after reranking
        top_k_reranked = 3
        final_context_docs = [doc for doc, score in reranked_docs_with_scores[:top_k_reranked]]
        context_str = "\n\n".join(final_context_docs)

        # 4. Generate the answer using Gemini
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
        
        return {
            "answer": response.text,
            "sources": final_context_docs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {str(e)}")

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)