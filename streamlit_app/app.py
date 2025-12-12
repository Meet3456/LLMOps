# streamlit_app/app.py
import requests
import streamlit as st

API_BASE = "http://localhost:8000"  # adjust if behind proxy / docker


def fetch_sessions():
    try:
        resp = requests.get(f"{API_BASE}/sessions", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def send_chat(session_id: str, message: str) -> str:
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"session_id": session_id, "message": message},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def upload_files(files) -> str:
    """
    Upload one or more files to /upload.
    Returns session_id created by backend.
    """
    files_payload = [("files", (f.name, f.getvalue())) for f in files]
    resp = requests.post(f"{API_BASE}/upload", files=files_payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"]


# -------------------- Streamlit layout -------------------- #

st.set_page_config(page_title="MultiDoc RAG Chat", layout="wide")
st.title("📚 Multi-Document RAG Chat")

# Sidebar: sessions + upload
st.sidebar.header("Sessions")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Button to refresh sessions
if st.sidebar.button("🔄 Refresh Sessions"):
    st.session_state["sessions"] = fetch_sessions()

sessions = st.session_state.get("sessions") or fetch_sessions()
st.session_state["sessions"] = sessions

# New upload
st.sidebar.subheader("New Document Session")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs / DOCX / TXT / CSV / HTML / Images",
    accept_multiple_files=True,
)

if st.sidebar.button("Create Session from Upload", type="primary") and uploaded_files:
    with st.spinner("Uploading and indexing..."):
        sid = upload_files(uploaded_files)
        st.session_state["session_id"] = sid
        st.session_state["message_history"] = []
        st.success(f"New session created: {sid}")

st.sidebar.markdown("---")
st.sidebar.subheader("Existing Sessions")

for s in sessions:
    label = f"{s['id']}"
    if st.sidebar.button(label):
        st.session_state["session_id"] = s["id"]
        st.session_state["message_history"] = []  # we don't pull full history here

# Main chat panel
if st.session_state["session_id"] is None:
    st.info("Create or select a session from the sidebar to start chatting.")
else:
    st.subheader(f"Session: {st.session_state['session_id']}")

    # Render local-only (Streamlit) history
    for msg in st.session_state["message_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question about the documents...")

    if user_input:
        # Show user message
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans = send_chat(st.session_state["session_id"], user_input)
                except Exception as e:
                    ans = f"Error from backend: {e}"

                st.markdown(ans)

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ans}
        )
