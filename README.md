Certainly! Here's the README content provided earlier, formatted correctly as a standalone Markdown document:

---

# LlamaIndex + Pinecone Document Querying System

This project provides a streamlined pipeline to process, embed, and query PDF documents using LlamaIndex and Pinecone. The system is designed to load documents from a directory, create embeddings, and retrieve relevant information based on a user's query. The pipeline has been optimized for speed and accuracy using timing decorators.

## Project Structure

```
.
├── embeddings/
│   ├── test.py            # Main script containing the LlamaIndex + Pinecone pipeline
├── .env                   # Environment variables (API keys, etc.)
├── pdf/                   # Directory containing PDF documents
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory and add your API keys:
```plaintext
HUGGINGFACE_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### 5. Add Your PDF Documents
Place the PDF documents you want to process in the `pdf/` directory.

## Usage

### 1. Running the Script

To run the script and start querying documents, use:
```bash
python embeddings/test.py
```

### 2. How It Works

- **Loading Environment Variables**: The script loads API keys and other necessary environment settings from the `.env` file.
- **Loading and Splitting PDF Documents**: Documents from the `pdf/` directory are loaded and split into smaller chunks for efficient processing.
- **Embedding Documents**: The system uses the Hugging Face model to create embeddings, which are then stored in a Pinecone index.
- **Querying the Documents**: The user can input a question, and the system will retrieve the most relevant documents, process them with LlamaIndex, and return a concise answer with source references.

### 3. Example Query

The following example demonstrates how to query the system:
```plaintext
Question: "What are the ownership of assets?"
```
Output:
```plaintext
LlamaIndex response with sources: The ownership of assets, as outlined in the Asset Handling Policy of INOVAQO, states that all assets and their properties are primarily owned by INOVAQO and are recorded in the Asset Register and books of accounts. While the company may hand over assets for use by its employees, those employees carry the responsibility of temporary ownership. The asset owner is responsible for ensuring that the asset and any associated data or information are classified and protected according to the Asset Handling Policy, as well as defining and periodically reviewing access controls, IT security, and compliance for electronic equipment.

(Source: Asset Handling Policy - INOVAQO' 23.pdf)
```

## Optimization

- **Timing Decorators**: The functions are wrapped in a timing decorator to measure and optimize the execution time.
- **Single Source References**: The script ensures that each document source is referenced only once in the final output to avoid redundancy.

## Dependencies

The project requires the following Python packages:
- `dotenv`
- `langchain`
- `langchain_g4f`
- `langchain_huggingface`
- `langchain_text_splitters`
- `pinecone-client`
- `pypdf`
- `llama_index`

These dependencies are included in the `requirements.txt` file and can be installed using `pip`.

## Contributing

Feel free to submit issues or pull requests to contribute to this project. Any enhancements, optimizations, or additional features are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This Markdown document is ready to be used as a README file without embedding it in any YAML or scripting context. Simply place it in your project directory as `README.md`.
