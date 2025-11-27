# Multipdf Chatbot using Gemini, FAISS, and Free Embeddings
# Goal: 
'''Build a chatbot that can answer questions based on multiple PDF 
documents using Google Gemini as the LLM, FAISS for vector
 storage, and free embeddings from Sentence Transformers.'''
# Import necessary libraries
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM + Prompt + LCEL
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import google.generativeai as genai

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=google_api_key)

# ------------------------------------
# Extract text from PDF files
# ------------------------------------
def get_pdf_text(paths):
    text = ""
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# -------------------------------------------------
# Split PDF text into chunks
# -------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

# ---------------------------------------------------------------
# Build vector embeddings & save FAISS index
# ---------------------------------------------------------------
def get_vector_store(text_chunks):
    os.makedirs("faiss_index", exist_ok=True)

    # FREE embeddings (no quota problems)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------------------------------------------------------
# Build RAG LCEL pipeline
# ---------------------------------------------------------------
def load_rag_chain(vector_store):

    prompt = PromptTemplate(
        template="""Use ONLY the following context to answer the question. 
If the answer is not in the context, say: "Answer not available in the context."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    retriever = vector_store.as_retriever()

    # Correct LCEL RAG Pipeline
    '''LCEL is a special syntax introduced by LangChain that allows you to build AI 
    workflows using pipe (|) operators — similar to Unix pipes.
It lets you chain:
LLMs
Prompts
Retrievers
into a single RAG pipeline with powerful features.'''
    rag_chain = (
        RunnableMap({
            "context": lambda q: "\n\n".join(
                d.page_content for d in retriever.get_relevant_documents(q)
            ),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    return rag_chain

# ---------------------------------------------------------------
# Chat with FAISS Database
# ---------------------------------------------------------------
def chat_with_pdf(question):
    if not os.path.exists("faiss_index/index.faiss"):
        return "❗ No index found. Upload PDFs and click 'Process PDFs' first."

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    rag_chain = load_rag_chain(db)
    response = rag_chain.invoke(question)

    return response.content


# ------------------------
# Streamlit UI
# ------------------------

st.title("💬 Multi-PDF Chatbot (Gemini + FAISS + Free Embeddings)")

with st.sidebar:
    st.header("📄 Upload PDFs")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if files:
            paths = []
            for file in files:
                with open(file.name, "wb") as f:
                    f.write(file.read())
                paths.append(file.name)

            text = get_pdf_text(paths)
            chunks = get_text_chunks(text)
            get_vector_store(chunks)

            st.success("✅ PDFs processed and indexed successfully!")
        else:
            st.error("Please upload at least one PDF first.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

st.header("Ask a Question About Your PDFs")
question = st.text_input("Type your question...")

if st.button("Submit") and question:
    answer = chat_with_pdf(question)
    st.session_state.history.append((question, answer))

# Display chat
for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
