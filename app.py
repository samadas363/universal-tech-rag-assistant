
import os
from pathlib import Path

import streamlit as st
import chromadb
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

APP_TITLE = "Universal Tech AI Privacy Assistant"
COLLECTION_NAME = "universal_tech_privacy"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "claude-sonnet-5-5"
)

TOP_K = 5

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. PROFESSIONAL STYLING
# =========================================================

st.markdown(
    """
    <style>
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    .profile-name {
        font-size: 30px;
        font-weight: 750;
        letter-spacing: -0.8px;
        line-height: 1.3;
        margin-bottom: 5px;
    }

    .profile-designation {
        font-size: 15px;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    .app-description {
        color: #64748b;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div[data-testid="stButton"] button {
        border-radius: 10px;
        min-height: 44px;
        text-align: left;
    }

    .footer-text {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        line-height: 1.8;
        padding-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD RAG COMPONENTS
# =========================================================

@st.cache_resource(show_spinner="Initializing AI assistant...")
def load_rag_system():

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is missing. "
            "Please check your .env file."
        )

    database_path = ROOT / "chroma_db"

    if not database_path.exists():
        raise FileNotFoundError(
            f"ChromaDB folder not found: {database_path}. "
            "Run ingest.py to create and populate the database."
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(
        path=str(database_path)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    claude_client = anthropic.Anthropic(
        api_key=api_key
    )

    return embedding_model, collection, claude_client


# =========================================================
# 4. RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_documents(
    question,
    embedding_model,
    collection
):

    if collection.count() == 0:
        return []

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_documents = []

    for index, document in enumerate(documents):

        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        retrieved_documents.append({
            "text": document,
            "metadata": metadata,
            "distance": distance
        })

    return retrieved_documents


# =========================================================
# 5. PREPARE DOCUMENT CONTEXT
# =========================================================

def build_context(retrieved_documents):

    if not retrieved_documents:
        return "No relevant document passages were retrieved."

    context_parts = []

    for index, item in enumerate(
        retrieved_documents,
        start=1
    ):

        metadata = item["metadata"]

        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("file_name")
            or "Unknown source"
        )

        section = metadata.get(
            "section",
            "Not specified"
        )

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Section: {section}

DOCUMENT CONTENT:
{item["text"]}
"""
        )

    return "\n\n".join(context_parts)


# =========================================================
# 6. GENERATE ANSWER USING CLAUDE
# =========================================================

def answer_question(
    question,
    retrieved_documents,
    claude_client,
    chat_history
):

    context = build_context(retrieved_documents)

    system_prompt = """
You are the Universal Tech AI Privacy Assistant.

Answer questions about the policies contained in the supplied
knowledge base.

IMPORTANT RULES:

1. Base factual answers on the supplied document passages.
2. Never invent policies, prices, deadlines, or contact details.
3. If the documents do not contain enough information, explain
   that the answer cannot be confirmed from the available sources.
4. Keep answers clear, helpful, and professional.
5. Distinguish explicit statements from interpretations.
6. Previous conversation may help interpret follow-up questions,
   but the retrieved documents are the factual evidence.
7. Do not claim that you have verified external websites.
8. The Universal Tech policy documents are sample content and
   must not be represented as independently verified real policies.
"""

    messages = []

    # Include recent conversation to support follow-up questions.
    for message in chat_history[-6:]:

        if message["role"] in ("user", "assistant"):

            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

    current_message = f"""
RETRIEVED DOCUMENT PASSAGES:

{context}

USER QUESTION:

{question}

Answer using the retrieved passages. If they do not provide
sufficient evidence, clearly explain the limitation.
"""

    messages.append({
        "role": "user",
        "content": current_message
    })

    response = claude_client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1200,
        system=system_prompt,
        messages=messages
    )

    answer = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )

    if not answer.strip():
        return "I couldn't generate an answer. Please try again."

    return answer


# =========================================================
# 7. DISPLAY SOURCE EVIDENCE
# =========================================================

def display_sources(retrieved_documents):

    if not retrieved_documents:
        st.caption(
            "No supporting document passages were retrieved."
        )
        return

    with st.expander(
        f"📄 View supporting sources "
        f"({len(retrieved_documents)} passages)",
        expanded=False
    ):

        st.caption(
            "Review the retrieved passages to check the evidence "
            "behind the answer."
        )

        for index, item in enumerate(
            retrieved_documents,
            start=1
        ):

            metadata = item["metadata"]

            source = (
                metadata.get("source")
                or metadata.get("filename")
                or metadata.get("file_name")
                or "Unknown source"
            )

            section = metadata.get(
                "section",
                "Not specified"
            )

            with st.container(border=True):

                st.markdown(
                    f"**{index}. {source}**"
                )

                st.caption(f"Section: {section}")

                if item["distance"] is not None:
                    st.caption(
                        f"Retrieval distance: "
                        f"{item['distance']:.4f}"
                    )

                st.write(item["text"])


# =========================================================
# 8. INITIALIZE SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# 9. PROFESSIONAL HEADER
# =========================================================

st.markdown(
    """
    <div class="profile-name">
        Abdul Samad
    </div>

    <div class="profile-designation">
        Data Analyst | AI Automation & RAG Developer
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

st.title("🤖 Universal Tech AI Privacy Assistant")

st.markdown(
    """
    <div class="app-description">
        Ask questions about privacy, refunds, shipping, payment
        security, and customer support. The assistant searches
        relevant policy documents and generates answers using
        their available information.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 10. CONNECT TO THE RAG SYSTEM
# =========================================================

try:

    (
        embedding_model,
        collection,
        claude_client
    ) = load_rag_system()

except Exception as error:

    st.error(
        "The AI assistant could not be initialized."
    )

    st.write(
        "Please check your API key, installed dependencies, "
        "ChromaDB folder, and collection configuration."
    )

    with st.expander("Technical error details"):
        st.code(str(error))

    st.stop()


# =========================================================
# 11. SUGGESTED QUESTIONS
# =========================================================

if not st.session_state.messages:

    st.markdown("#### How can I help you?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔒 Privacy policy",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What information does the privacy policy collect?"
            )

            st.rerun()

        if st.button(
            "📦 Shipping and returns",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is the shipping and returns policy?"
            )

            st.rerun()

    with col2:

        if st.button(
            "💳 Payment security",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What payment security measures are described?"
            )

            st.rerun()

        if st.button(
            "💬 Customer support",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "How can a customer contact support?"
            )

            st.rerun()


# =========================================================
# 12. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", [])
            )


# =========================================================
# 13. PROCESS QUESTIONS
# =========================================================

def process_question(question):

    question = question.strip()

    if not question:
        return

    previous_history = list(
        st.session_state.messages
    )

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching policy documents..."
            ):

                retrieved_documents = retrieve_documents(
                    question,
                    embedding_model,
                    collection
                )

                answer = answer_question(
                    question,
                    retrieved_documents,
                    claude_client,
                    previous_history
                )

            st.markdown(answer)

            display_sources(
                retrieved_documents
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved_documents
            })

        except Exception as error:

            error_message = (
                "I couldn't process your question. "
                "Please try again."
            )

            st.error(error_message)

            with st.expander("Technical error details"):
                st.code(str(error))

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })


# =========================================================
# 14. CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask a question about the policies..."
)

pending_question = st.session_state.pending_question

if pending_question:

    st.session_state.pending_question = None

    process_question(pending_question)

elif typed_question:

    process_question(typed_question)


# =========================================================
# 15. CLEAR CONVERSATION
# =========================================================

if st.session_state.messages:

    if st.button("Clear conversation"):

        st.session_state.messages = []
        st.session_state.pending_question = None

        st.rerun()


# =========================================================
# 16. PROFESSIONAL FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        <strong>Abdul Samad</strong><br>
        Data Analyst | AI Automation & RAG Developer
    </div>
    """,
    unsafe_allow_html=True
)
