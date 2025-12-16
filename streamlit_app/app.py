import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Doc Chat",
    layout="wide",
)

# ---------- CSS ----------
st.markdown(
    """
    <style>
    .chat-container { max-width: 900px; margin: auto; }
    .user-msg {
        background: #1f2937;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        font-size: 16px;
    }
    .ai-msg {
        background: #111827;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #e5e7eb;
        font-size: 16px;
    }
    textarea { font-size: 16px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
st.sidebar.title("💬 Sessions")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

sessions = requests.get(f"{API_BASE}/sessions").json()

for s in sessions:
    if st.sidebar.button(
        label=s["id"],
        key=f"session_btn_{s['id']}",
    ):
        st.session_state.session_id = s["id"]
        st.session_state.messages = []

if st.sidebar.button("➕ New Chat", key="new_chat_btn"):
    r = requests.post(f"{API_BASE}/session")
    st.session_state.session_id = r.json()["session_id"]
    st.session_state.messages = []

# ---------- MAIN ----------
st.title("📄 Multi-Document Chat")

if not st.session_state.session_id:
    st.info("Select or create a session")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

container = st.container()
with container:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(
                f"<div class='user-msg'>{m['content']}</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='ai-msg'>{m['content']}</div>", unsafe_allow_html=True
            )

query = st.text_area("Your question", height=90)

if st.button("Send") and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        r = requests.post(
            f"{API_BASE}/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": query,
            },
        )
        answer = r.json()["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
