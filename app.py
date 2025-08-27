import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Gemma 3 Chatbot with RAG")

hf_token = st.secrets["HF_API_TOKEN"]

@st.cache_resource
def get_rag_engine():
    return RAGEngine(hf_token)

rag_engine = get_rag_engine()

st.title("🤖 Gemma 3 Chatbot with RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            answer = rag_engine.query(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


