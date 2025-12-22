from datetime import datetime
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Document Chat",
    layout="wide",
)

# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
    .chat-message { font-size: 16px; line-height: 1.6; }
    .stChatMessage { padding: 12px; border-radius: 8px; }
    .stChatMessage.user { background-color: #f0f2f6; }
    .stChatMessage.assistant { background-color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# API HELPERS
# =========================================================
def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        # Some endpoints (like delete) might be 204 No Content
        if r.status_code == 204:
            return None
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")
        return None

# =========================================================
# STATE MANAGEMENT
# =========================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions" not in st.session_state:
    st.session_state.sessions = []

# =========================================================
# LOAD DATA
# =========================================================
def refresh_sessions():
    # Calling the endpoint moved to session.py
    data = api("GET", "/sessions")
    st.session_state.sessions = data if data else []

# Initial Load
refresh_sessions()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("💬 Sessions")

# Session List
for s in st.session_state.sessions:
    label = f"{s['id']}... • {s.get('ingestion_status', 'ready')}"
    if st.sidebar.button(label, key=f"sess-{s['id']}"):
        st.session_state.session_id = s["id"]
        st.session_state.messages = api("GET", f"/messages/{s['id']}")
        st.rerun()

st.sidebar.divider()

# New Chat
if st.sidebar.button("➕ New Chat", use_container_width=True):
    # This now hits the POST /session endpoint we added
    resp = api("POST", "/sessions")
    if resp:
        st.session_state.session_id = resp["session_id"]
        st.session_state.messages = []
        refresh_sessions()
        st.rerun()

# Delete Session
if st.session_state.session_id:
    if st.sidebar.button("🗑️ Delete Session", use_container_width=True):
        api("DELETE", f"/sessions/{st.session_state.session_id}")
        st.session_state.session_id = None
        st.session_state.messages = []
        refresh_sessions()
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📎 Uploaded Files")

# File Uploads
if st.session_state.session_id:
    files = api("GET", f"/files/{st.session_state.session_id}")
    if files:
        for f in files:
            st.sidebar.markdown(f"• {f['filename']}")
    else:
        st.sidebar.caption("No files uploaded")

    uploads = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True)
    
    if uploads:
        if st.sidebar.button("Process Files"):
            with st.spinner("Indexing documents..."):
                api(
                    "POST",
                    "/upload",
                    files=[("files", f) for f in uploads],
                    data={"session_id": st.session_state.session_id},
                )
            st.sidebar.success("Indexing completed")
            refresh_sessions()
            st.rerun()


# =========================================================
# MAIN CHAT
# =========================================================
st.title("📄 Multi-Document Chat")

if not st.session_state.session_id:
    st.info("👈 Create or select a session to start chatting.")
    st.stop()

# Render History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
query = st.chat_input("Ask about your documents...")

if query:
    # Optimistic UI update
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = api(
                "POST",
                "/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "message": query,
                },
            )
            
            if resp:
                answer = resp["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Failed to get response")