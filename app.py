import weaviate
import argparse
import timeit
import json
import time
import warnings
from llama_index import StorageContext, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import Ollama
import box
import yaml
import os


warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_documents(docs_path):
    documents = SimpleDirectoryReader(docs_path, required_exts=[".pdf"]).load_data()
    print(f"Loaded {len(documents)} documents")
    print(f"First document: {documents[0]}")
    return documents

def load_embedding_model(model_name):
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=model_name)
    )
    return embeddings

def build_index(weaviate_client, embed_model, documents, index_name):
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )

    return index

def get_rag_response(query, chain, debug=False):
    result = chain.query(query)

    try:
        # Convert and pretty print
        data = json.loads(str(result))
        data = json.dumps(data, indent=4)
        return data
    except (json.decoder.JSONDecodeError, TypeError):
        print("The response is not in JSON format.")
    return False

def main():
    # Load configuration
    with open('config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading documents...")
    documents = load_documents(cfg.DATA_PATH)

    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS)

    print("Building index...")
    index = build_index(client, embeddings, documents, cfg.INDEX_NAME)

    print("Index built successfully.")

    # Wait for user input for the query
    # query = input("Enter your query: ")
    query = os.getenv("QUERY", "Retrieve all available information!")

    print("Loading Ollama...")
    llm = Ollama(model=cfg.LLM, base_url=cfg.OLLAMA_BASE_URL, temperature=0)

    print("Building RAG pipeline...")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings)
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=cfg.INDEX_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    rag_chain = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context).as_query_engine(
        streaming=False,
        output_cls=None,  # Adjust as per your response class
        response_mode="compact"
    )

    print("Retrieving answer...")
    start = timeit.default_timer()
    answer = get_rag_response(query, rag_chain)
    end = timeit.default_timer()

    print(f'\nJSON answer:\n{answer}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")

if __name__ == "__main__":
    main()
