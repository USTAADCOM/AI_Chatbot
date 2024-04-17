"""
module to create embeddings of a document and generate response to user query.
"""
import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# Reranker Packages
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()
# Global variables
chat_history = []
openai_key = os.getenv("OPENAI_API_KEY")
default_persist_directory = 'CHROMA'
embedding = OpenAIEmbeddings(openai_api_key = openai_key)
llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key = openai_key, temperature = 0.6, max_tokens = 1000)
client = chromadb.PersistentClient(path = default_persist_directory)
output = client.list_collections()
# Extract names from each Collection object
names = [collection.name for collection in output]
collection_names = names
vector_db = Chroma(
            client = client,
            collection_name = collection_names[-1],
            embedding_function = embedding
            )
# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size = 1000, chunk_overlap = 100):
    """
    doc string
    """
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    print(pages)
    print("________________________________________________")
    doc_splits = text_splitter.split_documents(pages)
    print(doc_splits)
    return doc_splits

# Create vector database
def create_db(splits, collection_name):
    """
    doc string
    """
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents = splits,
        embedding = embedding,
        # client = new_client,
        persist_directory = default_persist_directory,
        collection_name = collection_name,
    )
    return vectordb
# Load vector database
def load_db():
    """
    doc string
    """
    if os.path.exists(default_persist_directory):
        return True
    else:
        return False

# Initialize langchain LLM chain
def initialize_llmchain(vector_db):
    """
    doc string
    """
    # HuggingFaceHub uses HF inference endpoints
    print("Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = 'answer',
        return_messages = True
    )
    print("Defining retrieval chain...")
    retriever = vector_db.as_retriever(search_kwargs = {"k": 10})
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever
    )
    qa_chain = ConversationalRetrievalChain.from_llm(llm = llm, 
                                                     memory = memory,
                                   retriever= compression_retriever,
                                   return_source_documents = True)
    print("Done!")
    return qa_chain
qa_chain = initialize_llmchain(vector_db)
# Initialize langchain LLM chain
def initialize_llmchain_after_refresh():
    """
    doc string
    """
    global vector_db
    global chat_history
    global qa_chain
    chat_history = []
    # HuggingFaceHub uses HF inference endpoints
    print("Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = 'answer',
        return_messages = True
    )
    print("Defining retrieval chain...")
    retriever = vector_db.as_retriever(search_kwargs = {"k": 10})
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever
    )
    qa_chain = ConversationalRetrievalChain.from_llm(llm = llm, 
                                                     memory = memory,
                                   retriever= compression_retriever,
                                   return_source_documents = True)
    print(qa_chain.memory.buffer)
    print("Done!")
    return "success"

# Initialize database
def initialize_database(upload_folder, chunk_size = 1000, chunk_overlap = 100):
    """
    doc string
    """
    names = []
    list_file_path  = [os.path.join('uploads', filename) for filename in os.listdir(upload_folder)]
    # Create collection_name for vector database
    print("Creating collection name...")
    collection_name = Path(list_file_path[0]).stem
    ## Remove space
    collection_name = collection_name.replace(" ","-") 
    ## Limit lenght to 50 characters
    collection_name = collection_name[:50]
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name[0] = 'A'
    if not collection_name[-1].isalnum():
        collection_name[-1] = 'Z'
    # print('list_file_path: ', list_file_path)
    print('Collection name: ', collection_name)
    print("Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load vector database
    print("Generating vector database...")
    # check db Existence
    if load_db():
        print("in Loading Process")
        client = chromadb.PersistentClient(path = default_persist_directory)
        new_client = client.get_or_create_collection(collection_name)     
        vector_db = Chroma.from_documents(
            documents = doc_splits,
            embedding = embedding,
            client = client,
            persist_directory = default_persist_directory,
            collection_name = collection_name,
        )
        output = client.list_collections()
        # Extract names from each Collection object
        names = [collection.name for collection in output]
        print(names)
    else:
        # global vector_db
        print("in Creating Process")
        vector_db = create_db(doc_splits, collection_name)
        vector_db.persist()
        client = chromadb.PersistentClient(path = default_persist_directory)       
        output = client.list_collections()
        # Extract names from each Collection object
        names = [collection.name for collection in output]
        print(names)
    return names

def conversation(query):
    """
    doc string
    """
    global chat_history
    global qa_chain
    result = qa_chain({"question": query, 'chat_history': chat_history}, return_only_outputs = True)
    response_answer = result["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    chat_history += [(query, result["answer"])]
    return response_answer, chat_history


def upload_file(file_obj):
    """
    doc string
    """
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    # print(file_path)
    return list_file_path
