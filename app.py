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

    # INGEST PROCESS


    # Load configuration
    with open('config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading documents...")
    # documents = load_documents(cfg.DATA_PATH)
    documents = SimpleDirectoryReader(cfg.DATA_PATH, required_exts=[".pdf"]).load_data()
    print(f"Loaded {len(documents)} documents")
    print(f"First document: {documents[0]}")


    print("Loading embedding model...")
    embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS)
        )
    print("Building index...")
    # index = build_index(client, embeddings, documents, cfg.INDEX_NAME)
    service_context = ServiceContext.from_defaults(embed_model=embeddings, llm=None)
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=cfg.INDEX_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )

    print("Index built successfully.")

    # Wait for user input for the query
    # query = input("Enter your query: ")
    query = os.getenv("QUERY", "Retrieve all available information!")



    # MAIN LLM PROCESS

    print("Loading Ollama...")
    llm = Ollama(model=cfg.LLM, base_url=cfg.OLLAMA_BASE_URL, temperature=0)

    print("Building RAG pipeline...")


    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading Ollama...")
    llm = Ollama(model=cfg.LLM, base_url=cfg.OLLAMA_BASE_URL, temperature=0)


    print("Building index...")
    service_context = ServiceContext.from_defaults(
        chunk_size=cfg.CHUNK_SIZE,
        llm=llm,
        embed_model=embeddings
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store, service_context=service_context
    )


    # index = build_index(cfg.CHUNK_SIZE, llm, embeddings, client, cfg.INDEX_NAME)

    print("Constructing query engine...")

    rag_chain = index.as_query_engine(
        streaming=False,
        output_cls=InvoiceInfo,
        response_mode="compact"
    )


    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # rag_chain = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context).as_query_engine(
    #     streaming=False,
    #     output_cls=None,  # Adjust as per your response class
    #     response_mode="compact"
    # )



    step = 0
    answer = False
    while not answer:
        step += 1
        if step > 1:
            print('Refining answer...')
            # add wait time, before refining to avoid spamming the server
            time.sleep(5)
        if step > 3:
            # if we have refined 3 times, and still no answer, break
            answer = 'No answer found.'
            break
        print('Retrieving answer...')
        answer = get_rag_response(args.input, rag_chain, args.debug)

    end = timeit.default_timer()

    print(f'\nJSON answer:\n{answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")

if __name__ == "__main__":
    main()
