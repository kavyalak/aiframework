import streamlit as st
import fitz, docx, os, io, hashlib
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI

def _extract_text(file_bytes: bytes, ext: str) -> str:
    text = ""
    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for p in doc:
                text += p.get_text()
    elif ext == "docx":
        d = docx.Document(io.BytesIO(file_bytes))
        for para in d.paragraphs:
            text += para.text + "\n"
    return text.strip()

def doc_compare_agent_page():
    st.title("🆚 Document Comparison Agent")
    load_dotenv()

    # LLM defaults to gpt-5-mini env (temperature=1)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=1
    )

    # Session state init
    if "compare_prompt" not in st.session_state:
        try:
            with open("promptcompare.txt", "r", encoding="utf-8") as f:
                st.session_state.compare_prompt = f.read()
        except:
            st.session_state.compare_prompt = (
                "You are a precise contract comparison assistant. Compare the two documents.\n"
                "- Summarize key similarities and differences\n"
                "- Highlight wording changes that alter risk/obligation\n"
                "- Flag redlines on: liability caps, IP, confidentiality, governing law, indemnity, termination, audit, data protection\n"
                "- Output a concise, sectioned report with bullet points and a final risk summary"
            )
    for k, v in {
        "compare_chat_history": [],
        "compare_memory": ConversationBufferMemory(return_messages=True),
        "docA_text": "",
        "docB_text": "",
        "docA_name": "",
        "docB_name": "",
        "docA_sig": None,
        "docB_sig": None,
        "compare_pair_sig": None,
        "compare_force_rerun": False,
    }.items():
        st.session_state.setdefault(k, v)

    st.markdown("Files are processed **in-memory only** (not stored).")

    with st.sidebar:
        st.markdown("### 🔧 Comparison Prompt")
        new_prompt = st.text_area("Edit comparison prompt", value=st.session_state.compare_prompt, height=220)
        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("Update Prompt", key="update_compare_prompt"):
                st.session_state.compare_prompt = new_prompt
                st.session_state.compare_force_rerun = True
                st.success("Comparison prompt updated for this session.")
        with colp2:
            if st.button("Reload from promptcompare.txt", key="reload_compare_prompt"):
                try:
                    with open("promptcompare.txt", "r", encoding="utf-8") as f:
                        st.session_state.compare_prompt = f.read()
                    st.session_state.compare_force_rerun = True
                    st.success("Reloaded from promptcompare.txt.")
                except Exception as e:
                    st.error(f"Could not read promptcompare.txt: {e}")

    # Uploaders (auto-load on selection)
    col1, col2 = st.columns(2)
    with col1:
        upA = st.file_uploader("Document A (.pdf/.docx)", type=["pdf", "docx"], key="compareA")
    with col2:
        upB = st.file_uploader("Document B (.pdf/.docx)", type=["pdf", "docx"], key="compareB")

    # Auto-load A
    if upA:
        bA = upA.getvalue()
        sigA = f"{upA.name}:{len(bA)}:{hashlib.md5(bA).hexdigest()}"
        if st.session_state.docA_sig != sigA:
            st.session_state.docA_text = _extract_text(bA, upA.name.split(".")[-1].lower())
            st.session_state.docA_name = upA.name
            st.session_state.docA_sig = sigA
            st.session_state.compare_force_rerun = True  # trigger compare
            st.success(f"Loaded A: {upA.name}")

    # Auto-load B
    if upB:
        bB = upB.getvalue()
        sigB = f"{upB.name}:{len(bB)}:{hashlib.md5(bB).hexdigest()}"
        if st.session_state.docB_sig != sigB:
            st.session_state.docB_text = _extract_text(bB, upB.name.split(".")[-1].lower())
            st.session_state.docB_name = upB.name
            st.session_state.docB_sig = sigB
            st.session_state.compare_force_rerun = True  # trigger compare
            st.success(f"Loaded B: {upB.name}")

    if st.session_state.docA_name or st.session_state.docB_name:
        st.caption(
            f"Active: "
            f"{('**A:** ' + st.session_state.docA_name) if st.session_state.docA_name else ''}"
            f"{' | ' if (st.session_state.docA_name and st.session_state.docB_name) else ''}"
            f"{('**B:** ' + st.session_state.docB_name) if st.session_state.docB_name else ''}"
        )

    # Helper: run comparison once when inputs/prompt change
    def _maybe_compare():
        if not (st.session_state.docA_text and st.session_state.docB_text):
            return
        pair_sig = f"{st.session_state.docA_sig}|{st.session_state.docB_sig}"
        if st.session_state.compare_force_rerun or (st.session_state.compare_pair_sig != pair_sig):
            with st.spinner("Comparing..."):
                prompt = (
                    f"{st.session_state.compare_prompt.strip()}\n\n"
                    f"--- Document A ({st.session_state.docA_name}) ---\n{st.session_state.docA_text}\n\n"
                    f"--- Document B ({st.session_state.docB_name}) ---\n{st.session_state.docB_text}\n"
                )
                response = llm.predict_messages(messages=[HumanMessage(content=prompt)])
                st.session_state.compare_chat_history.append(response)
                st.session_state.compare_memory.chat_memory.messages.append(response)
                st.session_state.compare_pair_sig = pair_sig
                st.session_state.compare_force_rerun = False
                st.markdown("### 📋 Comparison Report")
                st.markdown(response.content, unsafe_allow_html=True)

    # Auto-compare if both docs ready, or prompt changed
    _maybe_compare()

    # Chat history
    st.markdown("### 💬 Chat")
    for msg in st.session_state.compare_chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Agent:** {msg.content}", unsafe_allow_html=True)

    # Follow-up questions about the two docs
    user_input = st.chat_input("Ask a question about the differences, clauses, risks…")
    if user_input:
        st.markdown(f"**You:** {user_input}", unsafe_allow_html=True)
        st.session_state.compare_chat_history.append(HumanMessage(content=user_input))
        st.session_state.compare_memory.chat_memory.messages.append(HumanMessage(content=user_input))

        with st.spinner("Thinking..."):
            context = (
                f"Use the following two documents as context.\n\n"
                f"--- Document A ({st.session_state.docA_name}) ---\n{st.session_state.docA_text}\n\n"
                f"--- Document B ({st.session_state.docB_name}) ---\n{st.session_state.docB_text}\n\n"
                f"Question: {user_input.strip()}"
            )
            response = llm.predict_messages(messages=[HumanMessage(content=context)])
            st.session_state.compare_chat_history.append(response)
            st.session_state.compare_memory.chat_memory.messages.append(response)
            st.markdown(f"**Agent:** {response.content}", unsafe_allow_html=True)
