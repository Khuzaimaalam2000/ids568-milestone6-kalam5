import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents(folder_path):
    docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())

    return docs


# -----------------------------
# CHUNK DOCUMENTS
# -----------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    return splitter.split_documents(docs)


# -----------------------------
# BUILD VECTOR STORE
# -----------------------------
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    vectorstore.persist()
    return vectorstore


# -----------------------------
# TEST RETRIEVAL
# -----------------------------
def test_retrieval(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    print("\nTOP RETRIEVED CHUNKS:\n")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:300]}\n")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("Loading documents...")
    docs = load_documents("docs")

    print("Chunking documents...")
    chunks = chunk_documents(docs)

    print(f"Total documents: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")

    print("Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print("\nTesting retrieval...")
    test_query = "What is RAG?"
    test_retrieval(vectorstore, test_query)

    print("\nDONE: Chroma DB created at ./chroma_db")