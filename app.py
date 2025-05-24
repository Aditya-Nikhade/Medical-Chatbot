# --- LLM Provider Config ---
USE_OLLAMA = True  # Set to False to use OpenAI (requires API key)
OLLAMA_MODEL = "mistral"  # Or "llama2", "phi", etc.
OPENAI_MODEL = "gpt-3.5-turbo"
# --------------------------

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load and process documents
print("Loading PDF files...")
extracted_data = load_pdf_file(data='Data/')
if not extracted_data:
    print("[WARNING] No PDF files found in Data/. Please add medical PDFs for better chatbot performance.")
    exit(1)

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)

print("Initializing embeddings model...")
embeddings = download_hugging_face_embeddings()

# Pinecone index configuration
index_name = "medichatbot"
dimension = 384  # Dimension for all-MiniLM-L6-v2 model

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Using existing index: {index_name}")

# Batch processing for Pinecone upserts
BATCH_SIZE = 100
print(f"\nUploading documents in batches of {BATCH_SIZE}...")

docsearch = None
for i in range(0, len(text_chunks), BATCH_SIZE):
    batch = text_chunks[i:i+BATCH_SIZE]
    if i == 0:
        # Create the vector store with the first batch
        docsearch = PineconeVectorStore.from_documents(
            documents=batch,
            index_name=index_name,
            embedding=embeddings,
        )
    else:
        # Add subsequent batches
        docsearch.add_documents(batch)
    print(f"Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} documents)")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

# --- LLM Selection ---
if USE_OLLAMA:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.4)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.4)
# ---------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    try:
        # Retrieve context for logging
        retrieved_docs = retriever.get_relevant_documents(msg)
        print("\n[DEBUG] Retrieved context for query:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"Chunk {i}: {doc.page_content[:300]}\n---")
        # Use the RAG chain as before
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "").strip()
        # Filter out prompt structure or meta-messages
        if (not answer or
            "context provided" in answer.lower() or
            "system:" in answer.lower() or
            "question:" in answer.lower() or
            "answer:" in answer.lower() or
            answer.startswith("You are") or
            answer.startswith("The context") or
            answer.startswith("System:")):
            return "I'm sorry, I don't know the answer based on the provided information."
        return answer
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong. Please try again."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)