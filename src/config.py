import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# LLM Configuration
USE_OLLAMA = True
OLLAMA_MODEL = "mistral"  # Options: "mistral", "llama2", "phi"
OPENAI_MODEL = "gpt-3.5-turbo"  # Options: "gpt-3.5-turbo", "gpt-4"

# Vector Store Configuration
INDEX_NAME = "medichatbot"
BATCH_SIZE = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
METRIC = "cosine"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8080
DEBUG = True

# File Paths
DATA_DIR = "Data"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

# Logging Configuration
LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'
LOG_LEVEL = 'INFO' 