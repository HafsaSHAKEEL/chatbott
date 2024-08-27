import os
import re
import warnings
import time
from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_g4f import G4FLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from llama_index.legacy import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, set_global_service_context
from langchain.schema import Document
from langchain_pinecone import Pinecone as PC

from g4f import models

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r'clean_up_tokenization_spaces was not set')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def timing_decorator(func):
    """Decorator to time the execution of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


@timing_decorator
def load_environment_variables():
    """
    Load environment variables from a .env file.

    Returns:
        dict: Contains API keys and environment settings.
    """
    load_dotenv()
    return {
        "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT")
    }


@timing_decorator
def check_directory_exists(directory_path):
    """
    Check if a directory exists; raise an error if not.

    Args:
        directory_path (str): Path to check.

    Raises:
        ValueError: If the directory does not exist.
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist.")


@timing_decorator
def preprocess_document(text):
    """
    Remove specific patterns from document text.

    Args:
        text (str): Original document text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'P a g e \| \d+', '', text)
    text = re.sub(r'(Management reserves all the rights to change or eliminate any policy at any time)', '', text,
                  flags=re.IGNORECASE)
    return text


@timing_decorator
def load_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: Document objects containing extracted text.
    """
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return [Document(page_content=text, metadata={"source": file_path})]


@timing_decorator
def load_documents_from_directory(pdf_directory):
    """
    Load and split documents from PDF files in a directory.

    Args:
        pdf_directory (str): Directory containing PDF files.

    Returns:
        list: List of Document objects with text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    documents = []
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, file_name)
            pdf_documents = load_pdf(file_path)
            for document in pdf_documents:
                documents.extend(text_splitter.split_documents([document]))
    return documents


@timing_decorator
def get_pinecone_db(index_name, embeddings, pdf_dir):
    """
    Initialize or load a Pinecone index and load documents into it.

    Args:
        index_name (str): Name of the Pinecone index.
        embeddings (HuggingFaceEmbeddings): Embedding model.
        pdf_dir (str): Directory with PDF documents.

    Returns:
        PC: Pinecone instance representing the index.
    """
    pinecone_instance = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pine_indexes = pinecone_instance.list_indexes().names()
    if index_name not in pine_indexes:
        print(f"Creating index: {index_name}")
        docs = load_documents_from_directory(pdf_dir)
        pinecone_instance.create_index(
            name=index_name,
            dimension=768,
            metric='euclidean',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        return PC.from_documents(docs, embeddings, index_name=index_name)
    else:
        return PC.from_existing_index(index_name, embeddings)


@timing_decorator
def setup_retriever(vectorstore):
    """
    Set up a SelfQueryRetriever with a specified vector store.

    Args:
        vectorstore (VectorStore): The vector store used for retrieval.

    Returns:
        SelfQueryRetriever: Configured retriever.
    """
    document_content_description = "Policy-making document content"
    metadata_field_info = [AttributeInfo(name="source", description="The source of the document", type="string")]
    llm = G4FLLM(model=models.default)
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        verbose=True,
    )


@timing_decorator
def generate_prompt(question, context):
    """
    Generate a prompt for querying the HRMS system.

    Args:
        question (str): The user's question.
        context (str): The document context for answering the question.

    Returns:
        str: Formatted prompt string.
    """
    return f""" 
        You are an AI assistant for a Human Resource Management System (HRMS), responsible for delivering clear and precise answers about company policies based on the provided context. Your responses should be thorough, professional, and maintain an approachable tone.

        **Guidelines:**
        1. Context-Based Responses: Always base your answer strictly on the provided context to ensure relevance.
        2. Completeness: Make sure your answers are comprehensive and coherent. Avoid including incomplete or broken sentences.
        3. Accuracy: Ensure that the information provided is accurate and aligns with the context.
        4. Relevance: If the context doesn’t cover the question, provide the most relevant information available or suggest asking a more specific question.
        5. Clarity: Maintain clear and concise language throughout your response to avoid any ambiguity.
        6. Tone: Keep your tone professional yet approachable, ensuring a human-like interaction.
        7. Handling Greetings: Respond appropriately and helpfully to simple greetings or casual inquiries without deviating from the context.
        8. Word Limit: Keep your response concise, with a maximum of 400 characters, unless the question requires a more detailed explanation.
        9. Format: Structure your response in a single, well-organized paragraph, making it easy to read.
        10. Language: Always respond in clear and correct English, avoiding jargon unless necessary.

        Context: {context}
        User Input: {question}
        """


@timing_decorator
def answer_retriever(retriever, question):
    """
    Retrieve an answer from the retriever based on the user's question.

    Args:
        retriever (SelfQueryRetriever): The configured retriever.
        question (str): The user's question.

    Returns:
        dict or None: The retrieved answer, or None if an error occurs.
    """
    try:
        return retriever.invoke(input=question)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None


@timing_decorator
def run_llama_index_processing(pdf_dir, question):
    """
    Run the LlamaIndex processing pipeline, which includes loading documents, creating embeddings,
    and retrieving answers to the user's question.

    Args:
        pdf_dir (str): The directory containing PDF files.
        question (str): The user's question.

    Returns:
        None
    """
    # Load environment variables and check directory
    load_environment_variables()
    check_directory_exists(pdf_dir)

    # Set model and index details
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Model for embedding
    index_name = "document-embeddings"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create embeddings using Pinecone and load the documents
    pinecone_db = get_pinecone_db(index_name, embeddings, pdf_dir)

    # Process documents using LlamaIndex with the same Pinecone embeddings
    documents = SimpleDirectoryReader(pdf_dir).load_data()
    if not documents:
        print("No documents found in the directory.")
        return

    llm = G4FLLM(model=models.default)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
    set_global_service_context(service_context)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Perform the query and manually collect source references
    response = query_engine.query(question)

    if response:
        # Build the response with manually highlighted sources
        response_text = response.response.strip()
        retrieved_docs = response.source_nodes  # Accessing source nodes

        # Collect unique sources
        unique_sources = {doc.node.extra_info["file_name"] for doc in retrieved_docs if "file_name" in doc.node.extra_info}

        # Format source references
        source_references = "\n".join(f"(Source: {source})" for source in unique_sources)
        full_response = f"{response_text}\n\n{source_references}"
        print("LlamaIndex response with sources:", full_response)
    else:
        print("No relevant documents were found for your query.")


def main():
    """
    Main function to execute the query process and handle user interaction.

    Runs queries on the Pinecone database and performs additional processing with LlamaIndex.
    """
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pdf")
    question = "what are ownership of assets?"

    run_llama_index_processing(pdf_dir, question)


if __name__ == "__main__":
    main()
