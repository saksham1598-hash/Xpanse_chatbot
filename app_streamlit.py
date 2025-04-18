import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from config import RAG_CONFIG
from rag.rag_chain import RAGChain

custom_css = """
<style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
    }
    .custom-title {
        font-size: 2rem;
        font-weight: 700;
        color: #4b0082;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.2rem 0 1rem 0;
    }
    .custom-title span.icon {
        font-size: 2.2rem;
        color: #a020f0;
        font-weight: 900;
    }
    .source-card {
        background: #f4f6f8;
        border-left: 4px solid #4a90e2;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 6px 0 12px 0;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .source-card h4 {
        margin: 0 0 5px 0;
        color: #002a5c;
        font-size: 1rem;
    }
    /* Updated style for the snippet to allow scrolling */
    .source-snippet {
        margin: 5px 0;
        color: #333333;
        max-height: 150px;
        overflow-y: auto;
        padding-right: 5px; /* optional, for scrollbar spacing */
    }
    .score-details {
        background: #e3eaf2;
        font-family: monospace;
        font-size: 0.8rem;
        padding: 4px 6px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 4px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<div class="custom-title"><span class="icon">X</span> Xpanse Escrow Chatbot</div>', unsafe_allow_html=True)

# --- INIT SESSION STATE ---
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGChain(RAG_CONFIG)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR HISTORY ---
with st.sidebar:
    st.title("Conversation History")
    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages, start=1):
            role = msg["role"].capitalize()
            content = msg["content"]
            st.markdown(f"**{idx}. {role}:** {content}")
    else:
        st.info("No conversation yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
user_input = st.chat_input("Hi! I'm Xpanse Escrow Chatbot. Ask me about Escrow Accounts :")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- RAG PIPELINE RESPONSE ---
    answer = st.session_state.rag_pipeline.answer_question(user_input)

    try:
        sources_with_scores = st.session_state.rag_pipeline.retriever.get_relevant_documents_with_scores(user_input)
    except Exception as e:
        st.error(f"Error retrieving sources: {e}")
        sources_with_scores = []

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Show All Sources"):
        st.markdown("### Retrieved Sources")
        if sources_with_scores:
            for idx, (doc, score_details) in enumerate(sources_with_scores, start=1):
                # Instead of truncating the source, we now display the full content inside a scrollable div.
                full_content = doc.page_content
                st.markdown(f"""
                <div class="source-card">
                    <h4>Source {idx}</h4>
                    <div class="source-snippet">{full_content}</div>
                    <div class="score-details">Score: {score_details}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("No sources found.")
