import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.legacy import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.legacy.indices.vector_store import VectorIndexRetriever
from llama_index.legacy.postprocessor import SimilarityPostprocessor
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.vector_stores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    return os.getenv("PINECONE_API_KEY")

def check_directory_exists(directory_path):
    """Check if the given directory exists."""
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist.")

def load_documents_from_directory(directory_path):
    """Load PDF files from the specified directory."""
    reader = SimpleDirectoryReader(input_dir=directory_path)
    return reader.load_data()

def split_documents_into_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Chunk documents into smaller segments."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.text)
        for i, chunk in enumerate(chunks):
            source = doc.metadata.get('source', 'unknown_source')
            chunked_documents.append(Document(
                text=chunk,
                metadata={'source': f"{source}_chunk_{i}"}
            ))
    return chunked_documents

def initialize_pinecone_client(api_key):
    """Initialize Pinecone client."""
    return Pinecone(api_key=api_key)

def create_or_connect_index(pinecone_client, index_name):
    """Create or connect to a Pinecone index."""
    indexes = pinecone_client.list_indexes().names()
    if index_name not in indexes:
        print(f"Creating index '{index_name}'...")
        pinecone_client.create_index(
            name=index_name,
            dimension=768,
            metric="euclidean",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print(f"Index '{index_name}' already exists.")
    return pinecone_client.Index(index_name)

def print_pinecone_stats(pinecone_index, prefix=""):
    """Print Pinecone index stats with an optional prefix."""
    print(f"{prefix} Pinecone index stats:")
    try:
        stats = pinecone_index.describe_index_stats()
        print(stats)
    except Exception as e:
        print(f"Error retrieving Pinecone index stats: {e}")

def add_documents_to_pinecone(index, chunked_documents):
    """Add documents to the Pinecone index."""
    vector_store = PineconeVectorStore(pinecone_index=index)
    print("Adding documents to Pinecone index...")
    try:
        vector_store_index = VectorStoreIndex.from_documents(chunked_documents, vector_store=vector_store, show_progress=True)
        print("VectorStoreIndex created successfully.")
        return vector_store_index
    except Exception as e:
        print(f"Error during VectorStoreIndex creation: {e}")
        return None

def setup_query_engine(vector_store_index):
    """Set up the query engine with retriever and postprocessor."""
    retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=4)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
    query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])
    return query_engine

def query_pinecone_index(query_engine, query_text):
    """Query the Pinecone index using the query engine."""
    response = query_engine.query(query_text)
    return response

def main():
    # Load environment variables and Pinecone API key
    pinecone_api_key = load_environment_variables()

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    pdf_dir = os.path.join(parent_dir, "pdf")

    # Check directory existence
    check_directory_exists(pdf_dir)

    # Load documents and split into chunks
    documents = load_documents_from_directory(pdf_dir)
    print(f"Loaded {len(documents)} documents")
    chunked_documents = split_documents_into_chunks(documents)
    print(f"Total chunks created: {len(chunked_documents)}")

    # Initialize Pinecone client and create or connect to index
    pinecone_client = initialize_pinecone_client(pinecone_api_key)
    index_name = "llama-index"
    pinecone_index = create_or_connect_index(pinecone_client, index_name)

    # Print Pinecone index stats before adding documents
    print_pinecone_stats(pinecone_index, "Before indexing:")

    # Add documents to Pinecone index
    vector_store_index = add_documents_to_pinecone(pinecone_index, chunked_documents)

    # Print Pinecone index stats after adding documents
    print_pinecone_stats(pinecone_index, "After indexing:")

    # Set up query engine
    if vector_store_index:
        query_engine = setup_query_engine(vector_store_index)

        # Query Pinecone index
        response = query_pinecone_index(query_engine, "What is policy-based criteria?")
        print("Final Response:")
        print(response)

if __name__ == "__main__":
    main()
