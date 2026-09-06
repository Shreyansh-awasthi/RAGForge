import uuid
import time
import requests
import streamlit as st

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/query/stream"

st.set_page_config(page_title="RAGForge — Document Intelligence", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --paper: #F5F2EA;
        --panel: #FFFFFF;
        --ink: #1F2937;
        --inkdim: #6B7280;
        --accent: #B8862E;
        --accent-soft: #F0E6D2;
        --rule: #E4DFD0;
        --brick: #B4433D;
        --success: #3E7A52;
    }

    #MainMenu, header, footer { visibility: hidden; height: 0; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 980px; }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }
    .stApp { background-color: var(--paper); }

    /* ---------- Header ---------- */
    .folio-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 1.5rem;
        margin-bottom: 1.75rem;
    }
    .folio-brand { display: flex; align-items: baseline; gap: 0.6rem; }
    .folio-wordmark {
        font-family: 'Fraunces', serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    .folio-tagline { color: var(--inkdim); font-size: 0.85rem; }
    .folio-status {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        color: var(--inkdim);
        font-family: 'IBM Plex Mono', monospace;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background-color: var(--success);
        box-shadow: 0 0 0 3px rgba(62,122,82,0.15);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--rule);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .sidebar-title {
        font-family: 'Fraunces', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: var(--inkdim);
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    .sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--inkdim);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.5rem 0 0.5rem 0;
    }
    .doc-status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .doc-status-chip.ready {
        background-color: rgba(62,122,82,0.1);
        color: var(--success);
    }
    .doc-status-chip.empty {
        background-color: rgba(107,114,128,0.1);
        color: var(--inkdim);
    }

    /* ---------- Chat ---------- */
    .chat-row {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1.25rem;
        align-items: flex-start;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }

    .avatar {
        width: 28px; height: 28px;
        border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        flex-shrink: 0;
    }
    .avatar-user { background-color: var(--ink); color: #FFFFFF; }
    .avatar-assistant { background-color: var(--accent-soft); color: var(--accent); border: 1px solid var(--rule); }

    .bubble-wrap { max-width: 72%; }
    .bubble {
        padding: 0.7rem 1rem;
        border-radius: 12px;
        line-height: 1.65;
        white-space: pre-wrap;
        font-size: 0.93rem;
    }
    .bubble-user {
        background-color: var(--ink) !important;
        color: #FFFFFF !important;
        border-bottom-right-radius: 3px;
    }
    .bubble-assistant {
        background-color: var(--panel) !important;
        color: #111827 !important;
        border: 1px solid var(--rule);
        border-bottom-left-radius: 3px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .bubble-assistant.error {
        background-color: #FDF2F1 !important;
        color: var(--brick) !important;
        border-color: var(--brick);
    }
    .bubble-loading {
        background-color: var(--panel) !important;
        color: var(--inkdim) !important;
        border: 1px solid var(--rule);
        font-style: italic;
        border-bottom-left-radius: 3px;
    }

    .chat-meta {
        color: var(--inkdim);
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 0.3rem;
    }
    .chat-meta.user { text-align: right; }
    .chat-meta.assistant { text-align: left; }

    .empty-state {
        color: var(--inkdim);
        text-align: center;
        padding: 3rem 1rem;
        border: 1px dashed var(--rule);
        border-radius: 8px;
        font-size: 0.9rem;
    }
    .empty-state-title {
        font-family: 'Fraunces', serif;
        font-size: 1.1rem;
        color: var(--ink);
        margin-bottom: 0.4rem;
    }

    /* ---------- Inputs ---------- */
    .stTextInput label, .stTextArea label,
    .stTextInput label p, .stTextArea label p {
        color: var(--ink) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        opacity: 1 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid var(--rule) !important;
        border-radius: 6px !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        padding: 0.6rem 0.75rem !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: var(--inkdim) !important;
        opacity: 0.65;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border: 1px solid var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(184,134,46,0.12) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton button, .stFormSubmitButton button {
        background-color: var(--ink);
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.55rem 1.4rem;
        transition: background-color 0.15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        background-color: var(--accent);
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        background-color: transparent;
        color: var(--ink);
        border: 1px solid var(--rule);
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: var(--accent-soft);
        border-color: var(--accent);
        color: var(--accent);
    }

    .composer-divider {
        border-top: 1px solid var(--rule);
        margin: 1.75rem 0 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []


def start_new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


with st.sidebar:
    st.markdown('<div class="sidebar-title">RAGForge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Point RAGForge at a local document and ask it anything. Answers are grounded strictly in what the file contains.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Document</div>', unsafe_allow_html=True)
    file_path = st.text_input(
        "Document path",
        placeholder=r"C:\Users\you\Documents\file.pdf",
        label_visibility="collapsed",
    )
    file_path = file_path.strip().strip('"').strip("'") if file_path else file_path

    if file_path:
        st.markdown(
            '<div class="doc-status-chip ready">● Document set</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="doc-status-chip empty">○ No document yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    st.button("Start a new conversation", on_click=start_new_conversation)

api_endpoint = DEFAULT_ENDPOINT

st.markdown(
    """
    <div class="folio-header">
        <div class="folio-brand">
            <div class="folio-wordmark">RAGForge</div>
            <div class="folio-tagline">Document Q&amp;A</div>
        </div>
        <div class="folio-status"><span class="status-dot"></span> Connected</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.history:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">No questions yet</div>
            Add a document path in the sidebar, then ask a question below to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for turn in st.session_state.history:
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="bubble-wrap">
                    <div class="bubble bubble-user">{turn['question']}</div>
                    <div class="chat-meta user">You</div>
                </div>
                <div class="avatar avatar-user">YOU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        error_class = " error" if turn.get("is_error") else ""
        meta_text = (
            f"RAGForge &middot; {turn['elapsed']:.2f}s"
            if not turn.get("is_error")
            else "RAGForge &middot; error"
        )
        st.markdown(
            f"""
            <div class="chat-row assistant">
                <div class="avatar avatar-assistant">R</div>
                <div class="bubble-wrap">
                    <div class="bubble bubble-assistant{error_class}">{turn['answer']}</div>
                    <div class="chat-meta assistant">{meta_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="composer-divider"></div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    question = st.text_area("Your question", placeholder="What does the author mean by an asset?", label_visibility="visible")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not file_path or not file_path.strip():
        st.error("Add the document path first.")
    elif not question or not question.strip():
        pass
    else:
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="bubble-wrap">
                    <div class="bubble bubble-user">{question.strip()}</div>
                    <div class="chat-meta user">You</div>
                </div>
                <div class="avatar avatar-user">YOU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        placeholder = st.empty()
        placeholder.markdown(
            """
            <div class="chat-row assistant">
                <div class="avatar avatar-assistant">R</div>
                <div class="bubble-wrap">
                    <div class="bubble bubble-loading">Generating answer, please wait...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        full_answer = ""
        is_error = False
        t0 = time.perf_counter()
        try:
            response = requests.post(
                api_endpoint,
                json={
                    "file_path": file_path.strip(),
                    "question": question.strip(),
                    "thread_id": st.session_state.thread_id,
                },
                stream=True,
                timeout=120,
            )
            if response.ok:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_answer += chunk
                        placeholder.markdown(
                            f"""
                            <div class="chat-row assistant">
                                <div class="avatar avatar-assistant">R</div>
                                <div class="bubble-wrap">
                                    <div class="bubble bubble-assistant">{full_answer}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                full_answer = f"Request failed ({response.status_code})"
                is_error = True
        except requests.exceptions.RequestException:
            full_answer = f"Couldn't reach the server at {api_endpoint}. Is it running?"
            is_error = True

        elapsed = time.perf_counter() - t0

        st.session_state.history.append({
            "question": question.strip(),
            "answer": full_answer,
            "elapsed": elapsed,
            "is_error": is_error,
        })
        st.rerun()