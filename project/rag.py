import streamlit as st
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 RAG Chatbot")

st.write(
    "Retrieval-Augmented Generation using "
    "Streamlit and Ollama"
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


# ============================================================
# GET EMBEDDING USING OLLAMA
# ============================================================

def get_embedding(text):

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    return response["embedding"]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = sum(
        x * x
        for x in a
    ) ** 0.5

    magnitude_b = sum(
        y * y
        for y in b
    ) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document(text):

    chunks = create_chunks(text)

    for chunk in chunks:

        embedding = get_embedding(
            chunk
        )

        st.session_state.documents.append(
            {
                "text": chunk,
                "embedding": embedding
            }
        )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question, top_k=3):

    question_embedding = get_embedding(
        question
    )

    results = []

    for document in st.session_state.documents:

        similarity = cosine_similarity(
            question_embedding,
            document["embedding"]
        )

        results.append(
            {
                "text": document["text"],
                "score": similarity
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_documents
):

    context = ""

    for document in retrieved_documents:

        context += (
            "\n\n" +
            document["text"]
        )

    prompt = f"""
You are a RAG-based AI assistant.

Answer the question using ONLY the
information provided in the context.

If the answer is not present in the
context, say:

"I could not find the answer in the document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )

    return response["response"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "LLM Model:"
    )

    st.code(
        LLM_MODEL
    )

    st.write(
        "Embedding Model:"
    )

    st.code(
        EMBED_MODEL
    )

    st.divider()

    st.write(
        f"📚 Chunks stored: "
        f"{len(st.session_state.documents)}"
    )

    if st.button("Clear Knowledge Base"):

        st.session_state.documents = []

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()


# ============================================================
# DOCUMENT INPUT
# ============================================================

st.header("📄 Add Document")

document_text = st.text_area(
    "Paste your document here",
    height=250,
    placeholder=(
        "Paste the content of your "
        "document here..."
    )
)


if st.button(
    "Add Document",
    type="primary"
):

    if document_text.strip():

        with st.spinner(
            "Creating embeddings..."
        ):

            add_document(
                document_text
            )

        st.success(
            "Document added successfully!"
        )

    else:

        st.warning(
            "Please enter some text."
        )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.header("💬 Ask Questions")


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask something about your document..."
)


if question:

    # Show user question

    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Check knowledge base

    if not st.session_state.documents:

        answer = (
            "Please add a document first."
        )

        with st.chat_message(
            "assistant"
        ):

            st.warning(answer)

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🔍 Retrieving relevant information..."
            ):

                retrieved_documents = retrieve(
                    question,
                    top_k=3
                )


            with st.spinner(
                "🤖 Ollama is generating the answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_documents
                )


            st.write(answer)


            # Show retrieved information

            with st.expander(
                "🔎 View Retrieved Context"
            ):

                for i, document in enumerate(
                    retrieved_documents
                ):

                    st.write(
                        f"### Chunk {i + 1}"
                    )

                    st.write(
                        document["text"]
                    )

                    st.write(
                        f"Similarity: "
                        f"{document['score']:.4f}"
                    )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )