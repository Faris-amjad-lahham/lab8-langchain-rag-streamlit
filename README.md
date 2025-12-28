# Lab 8 — LLM Application using LangChain (RAG + Streamlit)

This project is the practical task for **Lab 8: LLM Application using LangChain**.  
It implements a **Generative AI course study chatbot** using:
- **LangChain** (prompt templates + LCEL chaining)
- **RAG (Retrieval-Augmented Generation)** with **FAISS**
- **HuggingFace** offline model (**google/flan-t5-small**)
- **Streamlit** UI for interaction

The chatbot answers questions about the Generative AI course topics and improves reliability by grounding answers in retrieved context.

---

## Features

- ✅ Offline LLM inference (no API key required)
- ✅ Prompt Template + LCEL Chain (LangChain Expression Language)
- ✅ RAG Pipeline:
  - Documents → Chunking → Embeddings → FAISS → Retrieval → Grounded Answer
- ✅ Streamlit web interface
- ✅ Demo questions + screenshots

---

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain (core + community + text splitters)
- Hugging Face Transformers
- sentence-transformers embeddings
- FAISS vector store

---

## Project Structure

