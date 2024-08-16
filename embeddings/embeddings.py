import os
import logging
import PyPDF2
import time
import openai
from huggingface_hub import InferenceApi
from llama_index.legacy import Document, VectorStoreIndex
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# API keys and tokens
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_token = os.getenv('HUGGINGFACE_API_TOKEN')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_pinecone(api_key, index_name, dimensionality):
    """Initialize Pinecone and create an index if it doesn't exist."""
    logging.info("Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimensionality,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

def initialize_hf_client(repo_id, token):
    """Initialize the Hugging Face Inference API client."""
    logging.info("Initializing Hugging Face Inference API client...")
    return InferenceApi(repo_id=repo_id, token=token)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logging.info(f"Extracting text from {pdf_path}...")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_embeddings(hf_client, text, retries=5, initial_backoff=1):
    """Get embeddings from the Hugging Face Inference API."""
    backoff = initial_backoff

    if not isinstance(text, str) or not text.strip():
        logging.error("Invalid input: Text must be a non-empty string.")
        return None

    text = str(text).strip()
    logging.debug(f"Text before sending to API: '{text}'")

    for attempt in range(retries):
        try:
            response = hf_client(inputs=text)
            logging.debug(f"Full API Response: {response}")

            if isinstance(response, dict):
                embeddings = response.get('embeddings')
                if embeddings and isinstance(embeddings, list) and all(isinstance(x, float) for x in embeddings):
                    if len(embeddings) == 768:
                        return embeddings
                    else:
                        logging.error(f"Unexpected embedding dimension: {len(embeddings)}")
                        break
            elif isinstance(response, list) and all(isinstance(x, float) for x in response):
                if len(response) == 768:
                    return response
                else:
                    logging.error(f"Unexpected embedding dimension: {len(response)}")
                    break
            else:
                logging.error("Unexpected response format or missing embeddings")
                logging.debug(f"Received response: {response}")
                break

        except Exception as e:
            logging.error(f"Request failed: {e}")
            logging.debug(f"Exception details: {e}")
            time.sleep(backoff)
            backoff *= 2

    return None

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing non-alphanumeric characters."""
    logging.info("Preprocessing text...")
    text = text.lower().strip()
    text = ' '.join(word for word in text.split() if word.isalnum())
    logging.debug(f"Text after preprocessing: {text}")
    return text

def chunk_text(text, chunk_size=128):
    """Chunk text into smaller pieces of specified size."""
    logging.info("Chunking text into smaller pieces...")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf_files(pdf_directory, hf_client, index, dimensionality):
    """Process PDF files by extracting text, generating embeddings, and storing them in Pinecone."""
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    documents = []

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        processed_text = preprocess_text(text)
        chunks = chunk_text(processed_text)

        for i, chunk in enumerate(chunks):
            logging.info(f"Generating embeddings for chunk {i + 1}/{len(chunks)} of {pdf_file}...")
            if len(chunk.strip()) < 10:
                logging.warning(f"Skipping chunk due to insufficient length: {chunk}")
                continue

            embedding = get_embeddings(hf_client, chunk)
            if embedding is not None:
                if len(embedding) == dimensionality:  # Verify dimensionality
                    logging.info(f"Embedding for chunk {i + 1}/{len(chunks)} of {pdf_file} has correct dimensionality: {len(embedding)}")
                else:
                    logging.error(f"Embedding for chunk {i + 1}/{len(chunks)} of {pdf_file} has incorrect dimensionality: {len(embedding)}")
                    continue

                index.upsert(vectors=[(f"{os.path.basename(pdf_file)}_chunk_{i}", embedding)])

                doc = Document(content=chunk, metadata={"source": f"{os.path.basename(pdf_file)}_chunk_{i}"})
                documents.append(doc)

    logging.info("Creating VectorStoreIndex from documents...")
    llama_index = VectorStoreIndex(documents)
    return llama_index

def query_pinecone(index, query_embedding, top_k=5):
    """Query Pinecone to retrieve the top_k most relevant document chunks based on the query embedding."""
    logging.info("Querying Pinecone for the most relevant document chunks...")
    response = index.query(queries=[query_embedding], top_k=top_k, include_values=False)
    results = response['results'][0]['matches']
    return results

def get_query_embedding(hf_client, query_text, retries=10, initial_backoff=2):
    """Generate embeddings for the query text using the Hugging Face Inference API."""
    logging.info(f"Generating embeddings for the query: {query_text}...")
    backoff = initial_backoff

    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt + 1} to generate embeddings.")
            query_embedding = get_embeddings(hf_client, query_text)
            if query_embedding is None:
                logging.error("Failed to generate embeddings for the query.")
                time.sleep(backoff)
                backoff *= 2
                continue
            return query_embedding
        except Exception as e:
            logging.error(f"An error occurred while generating query embeddings: {e}")
            time.sleep(backoff)
            backoff *= 2

    logging.error("Query embedding generation failed after multiple attempts.")
    raise ValueError("Query embedding generation failed.")

def retrieve_documents(index, hf_client, query_text, top_k=5):
    """Retrieve relevant document chunks from Pinecone based on the query text."""
    logging.info(f"Retrieving documents relevant to query: '{query_text}'")
    try:
        query_embedding = get_query_embedding(hf_client, query_text)
        logging.debug(f"Query embedding: {query_embedding}")

        if query_embedding is None:
            logging.error("Query embedding is None.")
            return []

        results = query_pinecone(index, query_embedding, top_k=top_k)
        logging.debug(f"Query results: {results}")

        if not results:
            logging.info("No relevant documents found for the query.")
            return []

        logging.info(f"Found {len(results)} relevant documents:")
        for match in results:
            logging.info(f"Document ID: {match['id']}, Score: {match['score']}")

        return results
    except Exception as e:
        logging.error(f"An error occurred while retrieving documents: {e}")
        raise

def format_retrieved_documents(results, document_map):
    """Format the retrieved documents for display."""
    formatted_results = []
    for match in results:
        doc_id = match['id']
        score = match['score']
        chunk = document_map.get(doc_id)
        formatted_results.append({
            "Document ID": doc_id,
            "Score": score,
            "Content": chunk.content,
            "Metadata": chunk.metadata,
        })
    return formatted_results

if __name__ == "__main__":
    # Define paths and parameters
    pdf_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdf'))
    index_name = "policy-documents"
    dimensionality = 768
    repo_id = "sentence-transformers/bert-base-nli-mean-tokens"
    query_text = "Explain the policy regarding data privacy."
    top_k = 5

    # Initialize Pinecone and Hugging Face Inference API client
    index = initialize_pinecone(api_key=pinecone_api_key, index_name=index_name, dimensionality=dimensionality)
    hf_client = initialize_hf_client(repo_id=repo_id, token=huggingface_api_token)

    # Process PDF files to create and store embeddings in Pinecone
    llama_index = process_pdf_files(pdf_directory, hf_client, index, dimensionality)
    logging.info("PDF processing and embedding creation complete.")

    # Query Pinecone to retrieve relevant documents
    results = retrieve_documents(index, hf_client, query_text, top_k=top_k)

    # Map document IDs to the actual documents
    document_map = {doc.metadata['source']: doc for doc in llama_index.documents}
    formatted_results = format_retrieved_documents(results, document_map)

    # Display the results
    logging.info("Query processing complete. Displaying retrieved documents:")
    for result in formatted_results:
        logging.info(f"Document ID: {result['Document ID']}, Score: {result['Score']}")
        logging.info(f"Content: {result['Content']}")
        logging.info(f"Metadata: {result['Metadata']}")
