# //If you want to add more books //
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables
load_dotenv()

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

print("\nVector store initialization complete!")