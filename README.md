# 📘 Multi-PDF Chatbot Using Gemini, FAISS, and Free Embeddings

**1. Introduction**

The Multi-PDF Chatbot is an AI-powered Retrieval Augmented Generation (RAG) system that allows users to upload multiple PDF documents and ask natural language questions based on their content.
The system uses:

**Google Gemini 2.5 Flash** as the Large Language Model (LLM)

**FAISS** for vector storage and fast similarity search

**Sentence Transformers** (MiniLM) for free embeddings

**LangChain LCEL** for building the RAG pipeline

**Streamlit **for the user interface

This application is designed to automate **document comprehension, information retrieval, and Q&A over multiple PDFs.**

**2. Objective**

The main goals of this project are:

Enable users to upload and process multiple PDF files

Convert PDF content into semantic embeddings

Store vectors efficiently using FAISS

Retrieve relevant chunks based on user queries

Use Gemini to generate accurate and contextual answers

Provide an easy-to-use Streamlit interface

**3. Technologies Used**
| Component              | Technology                                 |
| ---------------------- | ------------------------------------------ |
| Programming Language   | Python                                     |
| LLM                    | Google Gemini 2.5 Flash                    |
| Embeddings             | Sentence Transformers – *all-MiniLM-L6-v2* |
| Vector Store           | FAISS                                      |
| Framework              | Streamlit                                  |
| PDF Reader             | PyPDF2                                     |
| RAG Pipeline           | LangChain LCEL                             |
| Environment Management | python-dotenv                              |

**4. System Architecture**
Step-by-Step Workflow
  
1. User uploads PDFs 
2. Extract text from PDFs
3. Split text into chunks
4. Generate embeddings using MiniLM
5. Store embeddings in FAISS
6. User asks a question
7. Retrieve relevant chunks
8. Send context + question to Gemini
9. Gemini generates final answer
10. Display answer in Chat UI

**RAG Components**

**Retriever** → FAISS similarity search

**Generator** → Gemini LLM

**Prompt Template** → Ensures grounded answers

**LCEL** → Chains retriever → prompt → LLM

****5. Key Features**
✔ Multi-PDF Upload**

Upload and process multiple PDF documents at once.

✔ **Text Chunking**

Splits PDF text into 2000-character chunks with 200 overlap for better retrieval accuracy.

✔ **Free Embeddings (No API Cost)**

Uses "sentence-transformers/all-MiniLM-L6-v2" — fast and free.

✔ **FAISS Vector Store**

Stores and retrieves vectors efficiently for semantic search.

✔ **Gemini-Powered Answers**

High-quality, context-grounded responses using Gemini 2.5 Flash.

✔**Persistent Chat History**

Allows interactive question-answering.

✔ **Grounded Answers**

If the answer is not found in the PDF context, chatbot responds:

"Answer not available in the context."

**6. Pipeline Explanation**
A. **PDF Processing**

Extract full text from each page using PyPDF2

Merge into one combined dataset

B. **Chunking**

Chunk size: 2000
Overlap: 200
**Why?**

Large enough for context

Small enough for efficient embedding

C. **Embeddings**

Sentence Transformer (MiniLM):

Great balance of speed + accuracy

No token cost

Works well with FAISS

D. **Vector Store**

FAISS index is saved locally:

/faiss_index/index.faiss

E. **Retrieval**

The retriever finds the top relevant chunks based on user query.

F.**LLM Generation**

Gemini generates the final answer using:

User question

Retrieved context

7. **Prompt Used**

The custom prompt ensures reliable, grounded answers:

Use ONLY the following context to answer the question.
If the answer is not in the context, say:
"Answer not available in the context."

Context:
{context}

Question: {question}

Answer:

**8. User Interface (Streamlit)**

Features:

Upload PDFs

Process & index them

Ask questions

View chat history

Smooth and simple UI

**9. Results**

The system successfully:

Processes multiple PDFs

Retrieves accurate information

Generates context-based answers

Handles long PDF files

Provides reliable RAG-based Q&A

Example:

Q: What is the refund policy mentioned?
A: Answer extracted from the PDF context.


If not found:

A: Answer not available in the context.

**10. Conclusion**

The Multi-PDF Chatbot demonstrates the power of combining:

Free Embeddings

FAISS Vector Search

Google Gemini

LangChain LCEL

**This system can be scaled into:**

Research assistants

Corporate document search engines

Legal/medical document analyzers

University PDF archives

Knowledge management tools

It is lightweight, cost-effective, and highly extensible.

**11. Future Enhancements**

Support for PDF OCR (scanned PDFs)

Multi-page chunk mapping

Citation-based answers

Voice-based PDF Q&A

Save chat history to database

Export results as PDF or text

