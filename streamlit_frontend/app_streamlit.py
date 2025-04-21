import streamlit as st
from pathlib import Path
import sys, os, asyncio
from concurrent.futures import ThreadPoolExecutor

# allow imports from project root
sys.path.append(str(Path(os.path.dirname(__file__)).parent))

from dotenv import load_dotenv
load_dotenv()

from config import RAG_CONFIG
from src.rag.rag_chain import RAGChain
from src.rag.graph_rag import get_kg_answer

# --- Custom CSS (same as before) ---
CUSTOM_CSS = """
<style>
    .main { background-color: #ffffff; color: #333333; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa; border-right: 1px solid #ddd;
    }
    .custom-title {
        font-size: 2rem; font-weight: 700; color: #4b0082;
        display: flex; align-items: center; gap: 0.6rem; padding: 0.2rem 0 1rem 0;
    }
    .custom-title span.icon {
        font-size: 2.2rem; color: #a020f0; font-weight: 900;
    }
    .source-card {
        background: #f4f6f8; border-left: 4px solid #4a90e2;
        border-radius: 6px; padding: 10px 12px; margin: 6px 0 12px 0;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
        font-size: 0.9rem; line-height: 1.4;
    }
    .source-card h4 { margin: 0 0 5px 0; color: #002a5c; font-size: 1rem; }
    .source-snippet { margin: 5px 0; color: #333333; max-height: 150px;
        overflow-y: auto; padding-right: 5px; }
    .score-details {
        background: #e3eaf2; font-family: monospace; font-size: 0.8rem;
        padding: 4px 6px; border-radius: 4px; display: inline-block;
        margin-top: 4px;
    }
    .kg-option {
        background-color: #f0e6ff; border: 1px solid #d9c3ff;
        border-radius: 8px; padding: 15px; margin: 15px 0;
    }
    .kg-answer {
        margin-top: 15px; padding: 10px; background-color: #f8f4ff;
        border-radius: 5px; border-left: 3px solid #7e57c2;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Page header
st.markdown(
    '<div class="custom-title"><span class="icon"></span> Xpanse Escrow Chatbot</div>',
    unsafe_allow_html=True
)

# Helper to run async functions in sync context
def run_async(fn, *args, **kwargs):
    with ThreadPoolExecutor() as ex:
        return ex.submit(asyncio.new_event_loop().run_until_complete,
                         fn(*args, **kwargs)).result()

# Session state init
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGChain(RAG_CONFIG)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "kg_answer" not in st.session_state:
    st.session_state.kg_answer = None
if "latest_trace_id" not in st.session_state:
    st.session_state.latest_trace_id = None

# Sidebar conversation history
with st.sidebar:
    st.title("Conversation History")
    if not st.session_state.messages:
        st.info("No conversation yet.")
    else:
        for i, msg in enumerate(st.session_state.messages, 1):
            st.markdown(f"**{i}. {msg['role'].capitalize()}:** {msg['content']}")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Hi! I'm Xpanse Escrow Chatbot. Ask me about Escrow Accounts:")
if user_input:
    st.session_state.kg_answer = None
    st.session_state.kg_cleared = False
    st.session_state.last_query = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Answer generation
    answer = st.session_state.rag_pipeline.answer_question(user_input)
    trace_id = st.session_state.rag_pipeline.last_trace_id
    st.session_state.latest_trace_id = trace_id
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# --- Post-answer components ---
if st.session_state.last_query:

    # 1. Feedback
    with st.expander("⭐ Rate this answer"):
        st.write("Help us improve by rating the assistant’s response.")
        score = st.slider("Your score (0–5)", 0, 5, 3, 1, key="user_score")
        comment = st.text_input("Optional comment", key="user_comment")
        if st.button("Submit Feedback", key="submit_feedback"):
            st.session_state.rag_pipeline.submit_score(score, comment)
            st.success("Thanks! Your feedback has been recorded.")

    # 2. Show Sources
    try:
        sources = st.session_state.rag_pipeline.retriever.get_relevant_documents_with_scores(st.session_state.last_query)
    except Exception as e:
        st.error(f"Error retrieving sources: {e}")
        sources = []

    with st.expander("Show All Sources"):
        if sources:
            for idx, (doc, score) in enumerate(sources, 1):
                try:
                    if isinstance(score, dict) and "score" in score:
                        score_val = float(score["score"])
                    else:
                        score_val = float(score)
                    score_str = f"{score_val:.4f}"
                except Exception:
                    score_str = str(score)

                st.markdown(f"""
                <div class="source-card">
                  <h4>Source {idx}</h4>
                  <div class="source-snippet">{doc.page_content}</div>
                  <div class="score-details">Score: {score_str}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("No sources found.")

    # 3. Knowledge Graph
    with st.expander("Knowledge Graph RAG", expanded=False):
        st.markdown("#### Want a detailed answer using Knowledge Graph relations?")
        if st.button("Generate Knowledge Graph Answer", key="kg_button"):
            with st.spinner("Generating answer from Knowledge Graph..."):
                st.session_state.kg_answer = run_async(get_kg_answer, st.session_state.last_query)

        if st.session_state.kg_answer:
            st.markdown('<div class="kg-answer">', unsafe_allow_html=True)
            st.markdown(st.session_state.kg_answer)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Clear KG Answer", key="clear_kg"):
                st.session_state.kg_answer = None