# this file is for retrieving, basically it takes a query, embeds it, does similarity search on the ChromaDB

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./vector_store", embedding_function= embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever