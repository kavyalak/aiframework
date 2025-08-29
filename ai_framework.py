import streamlit as st
from utils.blob_storage import upload_to_blob, list_blobs, download_blob
from utils.llm_helpers import get_llm
import fitz, docx

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

def extract_text_from_bytes(file_bytes, ext):
    text = ""
    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif ext == "docx":
        import io
        d = docx.Document(io.BytesIO(file_bytes))
        for para in d.paragraphs:
            text += para.text + "\n"
    return text.strip()

def ai_framework_page():
    st.title("🧩 AI Framework")
    from dotenv import load_dotenv; load_dotenv()

    # LLM model selection (set once per session)
    if "framework_model_choice" not in st.session_state:
        st.session_state.framework_model_choice = "gpt-5-mini"
    if "framework_llm" not in st.session_state:
        st.session_state.framework_llm = get_llm(st.session_state.framework_model_choice)

    st.subheader("Choose your Language Model")
    model_choice = st.selectbox(
        "Language Model",
        ["gpt-5-mini", "gpt-4.1", "gpt-4.0"],
        index=["gpt-5-mini", "gpt-4.1", "gpt-4.0"].index(st.session_state.framework_model_choice)
    )
    if st.button("Set Model for this Session"):
        st.session_state.framework_model_choice = model_choice
        st.session_state.framework_llm = get_llm(model_choice)
        st.success(f"Using model: {model_choice}")

    llm = st.session_state.framework_llm

    # Prompt config
    if "framework_prompt" not in st.session_state:
        st.session_state.framework_prompt = "You are a helpful AI agent."
    st.markdown("### 🔧 Prompt Configuration")
    new_prompt = st.text_area("Edit system prompt", value=st.session_state.framework_prompt, height=150)
    if st.button("Update Framework Prompt"):
        st.session_state.framework_prompt = new_prompt
        st.success("Framework prompt updated.")

    # Blob file upload
    st.markdown("### 📁 File Upload and Selection (Azure Blob Storage)")
    uploaded_file = st.file_uploader("Upload file (.pdf, .docx)", type=["pdf", "docx"])
    if uploaded_file and st.button("Upload to Azure Blob"):
        upload_to_blob(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} to Azure Blob Storage.")

    blobs = list_blobs()
    selected_blob = st.selectbox("Select a file from storage", blobs if blobs else ["No files"])
    file_text = ""
    if selected_blob and selected_blob != "No files":
        if st.button("Load Selected File"):
            file_bytes = download_blob(selected_blob)
            ext = selected_blob.split('.')[-1].lower()
            file_text = extract_text_from_bytes(file_bytes, ext)
            st.session_state.framework_file_text = file_text
            st.success(f"Loaded: {selected_blob}")
    if "framework_file_text" not in st.session_state:
        st.session_state.framework_file_text = ""

    # Chat UI and memory
    if "framework_chat_history" not in st.session_state:
        st.session_state.framework_chat_history = []
    if "framework_memory" not in st.session_state:
        st.session_state.framework_memory = ConversationBufferMemory(return_messages=True)

    st.markdown("### 💬 Chat")
    for msg in st.session_state.framework_chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Agent:** {msg.content}", unsafe_allow_html=True)

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.markdown(f"**You:** {user_input}", unsafe_allow_html=True)
        st.session_state.framework_chat_history.append(HumanMessage(content=user_input))
        st.session_state.framework_memory.chat_memory.messages.append(HumanMessage(content=user_input))

        with st.spinner("Thinking..."):
            prompt = (
                f"{st.session_state.framework_prompt.strip()}\n\n"
                f"Document content:\n{st.session_state.framework_file_text.strip()}\n\n"
                f"User question:\n{user_input.strip()}"
            )
            response = llm.predict_messages(messages=[HumanMessage(content=prompt)])
            st.session_state.framework_chat_history.append(response)
            st.session_state.framework_memory.chat_memory.messages.append(response)
            st.markdown(f"**Agent:** {response.content}", unsafe_allow_html=True)
