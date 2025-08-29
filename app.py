import streamlit as st
from nda_agent import nda_agent_page
from ai_framework import ai_framework_page
from doc_compare_agent import doc_compare_agent_page  # NEW

st.set_page_config(page_title="AI Agent Suite", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose your app",
    ("Home", "Spirax Legal NDA Agent", "Document Comparison Agent", "AI Framework"),  # NEW option
    index=0
)

if page == "Home":
    st.title("Welcome to AI Agent Suite")
    st.write(
        """
        Choose an agent from the sidebar:
        - **Spirax Legal NDA Agent**: Specialized NDA review with AI.
        - **Document Comparison Agent**: Upload two docs and compare differences with AI.
        - **AI Framework**: Upload, store, and analyze any PDF/DOCX file with your choice of language model.
        """
    )
elif page == "Spirax Legal NDA Agent":
    nda_agent_page()
elif page == "Document Comparison Agent":
    doc_compare_agent_page()  # NEW
elif page == "AI Framework":
    ai_framework_page()

