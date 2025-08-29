import streamlit as st
import fitz, docx, os, io, re, hashlib
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from utils.blob_storage import list_blobs, upload_to_blob

def _unique_filename(desired: str, existing: set[str]) -> str:
    """Return a unique 'name (n).ext' if desired already exists (never '(1)(1)')"""
    if desired not in existing:
        return desired
    base, ext = os.path.splitext(desired)
    pattern = re.compile(re.escape(base) + r" \((\d+)\)" + re.escape(ext) + r"$")
    nums = []
    for name in existing:
        if name == desired:
            nums.append(0)
        m = pattern.search(name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except:
                pass
    n = (max(nums) + 1) if nums else 1
    return f"{base} ({n}){ext}"

def _extract_text_from_bytes(file_bytes: bytes, ext: str) -> str:
    text = ""
    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif ext == "docx":
        document = docx.Document(io.BytesIO(file_bytes))
        for para in document.paragraphs:
            text += para.text + "\n"
    return text.strip()

def nda_agent_page():
    st.title("📄 Spirax NDA Review Agent")
    from dotenv import load_dotenv; load_dotenv()

    # LLM defaults to gpt-5-mini env (temperature=1)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=1
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prompt" not in st.session_state:
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                st.session_state.prompt = f.read()
        except:
            st.session_state.prompt = (
                "You are a helpful AI NDA review assistant. "
                "Analyze and interpret the contract based on user questions and identify any risks or red flags."
            )
    if "doc_text" not in st.session_state:
        st.session_state.doc_text = ""
    if "doc_display_name" not in st.session_state:
        st.session_state.doc_display_name = ""
    if "doc_file_sig" not in st.session_state:
        st.session_state.doc_file_sig = None

    with st.sidebar:
        st.markdown("### 🔧 Prompt Configuration")
        new_prompt = st.text_area("Edit system prompt", value=st.session_state.prompt, height=200)
        if st.button("Update Prompt"):
            st.session_state.prompt = new_prompt
            st.success("Prompt updated for this session.")

        uploaded_file = st.file_uploader(
            "Upload NDA Document (.pdf or .docx)", type=["pdf", "docx"], key="nda_uploader"
        )

        # Perform upload only on an explicit click to avoid Streamlit re-run duplicates
        if uploaded_file and st.button("Upload to Blob & Analyze", key="nda_upload_btn"):
            file_bytes = uploaded_file.getvalue()
            file_sig = f"{uploaded_file.name}:{len(file_bytes)}:{hashlib.md5(file_bytes).hexdigest()}"

            # prevent duplicate upload/analysis in this session
            if st.session_state.doc_file_sig == file_sig:
                st.info("This file is already uploaded and analyzed in this session.")
            else:
                # 1) resolve unique blob name
                existing_blob_names = set(list_blobs())  # add prefix if you use subfolders
                desired_name = uploaded_file.name
                unique_name = _unique_filename(desired_name, existing_blob_names)

                # 2) upload once to blob
                upload_to_blob(io.BytesIO(file_bytes), blob_name=unique_name)

                # 3) extract text for analysis
                ext = desired_name.split(".")[-1].lower()
                st.session_state.doc_text = _extract_text_from_bytes(file_bytes, ext)
                st.session_state.doc_display_name = unique_name
                st.session_state.doc_file_sig = file_sig

                st.success(f"Uploaded to Blob as '{unique_name}' and processed for analysis.")

    if st.session_state.doc_display_name:
        st.caption(f"Active document: **{st.session_state.doc_display_name}**")

    # Chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Agent:** {msg.content}", unsafe_allow_html=True)

    user_input = st.chat_input("Ask your NDA-related question...")
    if user_input:
        st.markdown(f"**You:** {user_input}", unsafe_allow_html=True)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.memory.chat_memory.messages.append(HumanMessage(content=user_input))

        with st.spinner("Analyzing..."):
            system_prompt = st.session_state.prompt
            file_context = st.session_state.doc_text

            full_prompt = (
                f"{system_prompt.strip()}\n\n"
                f"Document content:\n{file_context.strip()}\n\n"
                f"User question:\n{user_input.strip()}"
            )
            response = llm.predict_messages(messages=[HumanMessage(content=full_prompt)])
            st.session_state.chat_history.append(response)
            st.session_state.memory.chat_memory.messages.append(response)
            st.markdown(f"**Agent:** {response.content}", unsafe_allow_html=True)
