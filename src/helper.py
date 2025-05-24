import logging
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import DATA_DIR, EMBEDDING_MODEL, LOG_FORMAT, LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def load_pdf_file(data: str = DATA_DIR) -> List[Document]:
    """
    Load PDF files from the specified directory.
    
    Args:
        data (str): Path to directory containing PDF files
        
    Returns:
        List[Document]: List of loaded documents
        
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no PDF files are found
    """
    try:
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data}")
            
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No PDF files found in {data}")
            
        logger.info(f"Successfully loaded {len(documents)} PDF files")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading PDF files: {str(e)}")
        raise

def advanced_chunk_split(
    extracted_data: List[Document],
    max_length: int = 1000,
    overlap: int = 100
) -> List[Document]:
    """
    Split documents into chunks using an advanced strategy.
    First splits by paragraphs, then by size if needed.
    
    Args:
        extracted_data (List[Document]): List of documents to split
        max_length (int): Maximum chunk size
        overlap (int): Overlap between chunks
        
    Returns:
        List[Document]: List of document chunks
    """
    try:
        # Split by paragraphs first
        paragraph_chunks = []
        for doc in extracted_data:
            paragraphs = doc.page_content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    paragraph_chunks.append(Document(
                        page_content=para,
                        metadata=doc.metadata
                    ))
        
        # Further split long paragraphs
        final_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=overlap
        )
        
        for chunk in paragraph_chunks:
            if len(chunk.page_content) > max_length:
                final_chunks.extend(splitter.split_documents([chunk]))
            else:
                final_chunks.append(chunk)
                
        logger.info(f"Split {len(extracted_data)} documents into {len(final_chunks)} chunks")
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    Wrapper function for text splitting.
    
    Args:
        extracted_data (List[Document]): List of documents to split
        
    Returns:
        List[Document]: List of document chunks
    """
    return advanced_chunk_split(extracted_data)

def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """
    Download and initialize HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings model
        
    Raises:
        Exception: If model download fails
    """
    try:
        logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        raise