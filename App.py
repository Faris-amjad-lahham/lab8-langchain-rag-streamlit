import streamlit as st

# =========================
# LangChain + RAG Backend
# =========================

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------
# LLM
# -------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------
# Documents (RAG Knowledge)
# -------------------------
course_docs = [
    "Generative AI models learn data distributions to generate new content.",
    "Transformers rely on self-attention instead of recurrence.",
    "RAG reduces hallucination by grounding answers in retrieved documents.",
    "LangChain connects LLMs with prompts, memory, tools, and retrieval.",
    "Fine-tuning adapts pre-trained models to specific tasks."
]

documents = [Document(page_content=text) for text in course_docs]

# -------------------------
# Text Splitting
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

# -------------------------
# Embeddings + Vector Store
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor for a Generative AI course. Use the context to answer."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# -------------------------
# RAG Chain (LCEL)
# -------------------------
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Generative AI Study Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Generative AI Study Chatbot")
st.write("Ask questions about the Generative AI course.")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Why is RAG important in LLM applications?"
)

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(user_question)

        st.subheader("Answer")
        st.write(answer)

