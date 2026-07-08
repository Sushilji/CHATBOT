import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


def load_rag_components():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables or Streamlit secrets")

    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    document = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents([document])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return retriever, llm


@st.cache_resource(show_spinner=False)
def get_rag():
    return load_rag_components()


def build_prompt(question: str, retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def get_answer(llm, question: str, retriever):
    try:
        retrieved_docs = retriever.invoke(question)
    except AttributeError:
        retrieved_docs = retriever.get_relevant_documents(question)

    prompt = build_prompt(question, retrieved_docs)
    response = llm.invoke(prompt)
    answer = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
    return answer, retrieved_docs


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        .big-container {
            padding: 2rem 3rem 2rem 3rem;
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #f8fafc;
            border-radius: 20px;
        }
        .stTextArea>div>div>textarea {
            min-height: 200px;
            font-size: 1.05rem;
            border-radius: 16px;
            padding: 18px;
            background: #111827;
            color: #e2e8f0;
            border: 1px solid #334155;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
            color: white;
            font-weight: 700;
            border-radius: 14px;
            padding: 0.85rem 1.5rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.35);
        }
        .chat-card {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 18px;
            padding: 1.3rem;
            margin-bottom: 1rem;
        }
        .chat-label {
            color: #38bdf8;
            font-weight: 700;
        }
        .stAlert {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAG Chatbot")
    st.markdown("Ask questions about the text data stored in `data.txt`.")

    retriever, llm = get_rag()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.container():
        with st.form(key="query_form"):
            question = st.text_area(
                "Your question",
                placeholder="Type your question here and press Ask",
                key="question_input",
            )
            submit = st.form_submit_button("Ask")
            clear_chat = st.form_submit_button("Clear chat")

            if submit and question:
                with st.spinner("Working on your answer..."):
                    time.sleep(1.5)
                    answer, retrieved_docs = get_answer(llm, question, retriever)
                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "docs": retrieved_docs,
                    }
                )

            if clear_chat:
                st.session_state.history = []
                st.experimental_rerun()

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(
                f"<div class='chat-card'><div class='chat-label'>You</div><p>{item['question']}</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='chat-card'><div class='chat-label'>Bot</div><p>{item['answer']}</p></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Enter a question above and click Ask to query the documents.")


if __name__ == "__main__":
    main()
