import os
import re
import warnings
from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_g4f import G4FLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from llama_index.legacy import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, set_global_service_context, \
    Document
from langchain_pinecone import Pinecone as PC

from g4f import models

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r'`clean_up_tokenization_spaces` was not set')
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def check_directory_exists(directory_path):
    """
    Check if a directory exists; raise error if not.

    Args:
        directory_path (str): Path to check.
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist.")


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
    # Other similar cleaning operations...
    return text


def load_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: Document object containing extracted text.
    """
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return [Document(page_content=text, metadata={"source": file_path})]


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


def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split Document objects into smaller chunks.

    Args:
        documents (list): List of Document objects.

    Returns:
        list: List of chunked Document objects.
        :param documents:
        :param chunk_overlap:
        :param chunk_size:
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        chunked_documents.extend(
            Document(page_content=chunk,
                     metadata={'source': f"{doc.metadata.get('source', 'unknown_source')}_chunk_{i}"})
            for i, chunk in enumerate(chunks)
        )
    return chunked_documents


def get_pinecone_db(index_name, embeddings, pdf_dir):
    """
    Initialize or load a Pinecone index, and load documents into it.

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


def run_query(pdf_dir, question):
    """
    Run a query on the Pinecone database using LangChain.

    Args:
        pdf_dir (str): Directory with PDF documents.
        question (str): The user's question.

    Returns:
        dict or None: The retrieved answer.
    """
    env_vars = load_environment_variables()
    check_directory_exists(pdf_dir)

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    index_name = "document-embeddings"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    pinecone_db = get_pinecone_db(index_name, embeddings, pdf_dir)
    retriever = setup_retriever(pinecone_db)

    answer_dict = answer_retriever(retriever, question)
    return answer_dict


def run_llama_index_processing(pdf_dir, question):
    """
    Process documents using LlamaIndex and generate a response to the user's question.

    Args:
        pdf_dir (str): Directory containing PDF documents.
        question (str): The user's question.

    Returns:
        None: Prints the response or an error message.
    """
    documents = SimpleDirectoryReader(pdf_dir).load_data()
    if not documents:
        print("No documents found in the directory.")
        return

    llm = G4FLLM(model=models.default)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
    set_global_service_context(service_context)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    context = "\n".join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])
    prompt = generate_prompt(question, context)

    response = query_engine.query(prompt)
    if response:
        print("LlamaIndex response:", response.response)
    else:
        print("No response from LlamaIndex.")


def main():
    """
    Main function to execute the query process and handle user interaction.

    Runs queries on the Pinecone database and performs additional processing with LlamaIndex.
    """
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pdf")
    question = "what is policy of vacation?"


    run_llama_index_processing(pdf_dir, question)


if __name__ == "__main__":
    main()
