import streamlit as st
import ollama

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.write("Paste your context and ask questions based on it.")

# -------------------------------
# Context Input
# -------------------------------

st.subheader("📄 Paste your context here")

context = st.text_area(
    "Context",
    placeholder="Paste your paragraph or document text here...",
    height=250
)

# -------------------------------
# Question Input
# -------------------------------

st.subheader("❓ Ask your question")

question = st.text_input(
    "Question",
    placeholder="Ask something about the context..."
)

# -------------------------------
# Buttons
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    ask = st.button("🔍 Ask Question", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

# -------------------------------
# Generate Answer
# -------------------------------

if ask:

    if not context.strip():
        st.warning("⚠️ Please paste your context first.")

    elif not question.strip():
        st.warning("⚠️ Please enter a question.")

    else:

        prompt = f"""
You are a helpful RAG chatbot.

Answer the question using ONLY the context provided below.

If the answer is not available in the context, say:
"I don't know based on the provided context."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        with st.spinner("Thinking..."):

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        answer = response["message"]["content"]

        st.subheader("💡 Answer")
        st.success(answer)