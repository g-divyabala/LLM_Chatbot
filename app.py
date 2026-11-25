
import streamlit as st
import ollama
import csv
import time
import uuid

from pathlib import Path
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

VECTOR_DB_PATH = PROJECT_ROOT

LOGS_PATH = PROJECT_ROOT

LOG_FILE = LOGS_PATH / "chat_logs.csv"


# ==========================================
# 2. CREATE LOG FOLDER IF REQUIRED
# ==========================================

LOGS_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Knowledge Chatbot")

st.write("Ask questions based on the provided knowledge base.")


# ==========================================
# 4. CHECK FAISS DATABASE FILES
# ==========================================

FAISS_INDEX = VECTOR_DB_PATH / "index.faiss"
FAISS_METADATA = VECTOR_DB_PATH / "index.pkl"

if not FAISS_INDEX.exists() or not FAISS_METADATA.exists():
    st.error("FAISS database files are missing!")

    st.write("Expected database location:")
    st.code(str(VECTOR_DB_PATH))

    st.stop()


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==========================================
# 6. LOAD FAISS VECTOR DATABASE
# ==========================================

@st.cache_resource
def load_vector_database():

    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        embedding_model,
        index_name="index",
        allow_dangerous_deserialization=True
    )


try:

    vector_db = load_vector_database()

except Exception as e:

    st.error("Unable to load FAISS vector database.")
    st.exception(e)
    st.stop()


st.success("Knowledge base loaded successfully!")


# ==========================================
# 7. CREATE CHAT LOG CSV
# ==========================================

if not LOG_FILE.exists():

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "query_id",
            "timestamp",
            "question",
            "retrieved_context",
            "answer",
            "retrieval_time_seconds",
            "generation_time_seconds",
            "total_latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "feedback"
        ])


# ==========================================
# 8. CHAT SESSION
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# 9. USER QUESTION
# ==========================================

question = st.chat_input("Ask a question about Python...")


if question:

    query_id = str(uuid.uuid4())

    # Store user question

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.write(question)


    # ======================================
    # 10. DOCUMENT RETRIEVAL
    # ======================================

    try:

        retrieval_start = time.time()

        results = vector_db.similarity_search(
            question,
            k=2
        )

        retrieval_time = time.time() - retrieval_start

        context = "\n\n".join(
            result.page_content
            for result in results
        )

    except Exception as e:

        st.error("Document retrieval failed.")
        st.exception(e)
        st.stop()


    # ======================================
    # 11. CREATE PROMPT
    # ======================================

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


    # ======================================
    # 12. GENERATE ANSWER USING OLLAMA
    # ======================================

    try:

        generation_start = time.time()

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        generation_time = time.time() - generation_start

        answer = response["message"]["content"]

    except Exception as e:

        st.error("Ollama response generation failed.")
        st.exception(e)
        st.stop()


    # ======================================
    # 13. TOKEN USAGE
    # ======================================

    prompt_tokens = response.get(
        "prompt_eval_count",
        0
    )

    completion_tokens = response.get(
        "eval_count",
        0
    )

    total_tokens = prompt_tokens + completion_tokens


    # ======================================
    # 14. TOTAL LATENCY
    # ======================================

    total_latency = retrieval_time + generation_time


    # ======================================
    # 15. DISPLAY ANSWER
    # ======================================

    with st.chat_message("assistant"):

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # ======================================
    # 16. DISPLAY PERFORMANCE METRICS
    # ======================================

    st.subheader("Performance Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Retrieval Time",
        f"{retrieval_time:.2f}s"
    )

    col2.metric(
        "Generation Time",
        f"{generation_time:.2f}s"
    )

    col3.metric(
        "Total Latency",
        f"{total_latency:.2f}s"
    )


    # ======================================
    # 17. SAVE CHAT LOGS
    # ======================================

    try:

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                query_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                context,
                answer,
                round(retrieval_time, 4),
                round(generation_time, 4),
                round(total_latency, 4),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                "not_provided"
            ])

    except Exception as e:

        st.warning(f"Unable to save chat logs: {e}")
