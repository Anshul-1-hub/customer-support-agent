# for the project im treating the Streamlit Docs as the knowledge base, so in this file, I load the docs, split them into chunks, embed and then store it in a vector db(particularly, ChromaDB for this projectm, because it is local, whereas pinecone is cloud hosted), so basically a one-time setup file.

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# loading the docs
loader = WebBaseLoader("https://docs.streamlit.io")
docs = loader.load()

# chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# embedding and storing in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./vector_store")