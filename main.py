import streamlit as st
import os
from groq import Groq
import pandas as pd

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


# --- RAG/AGENT FUNCTIONS ---

@st.cache_data
def get_vector_store_from_file(file_path):
    """Loads a text-based file and creates a RAG-ready vector store."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.error("File type not supported for text analysis.")
        return None
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store

def create_rag_chain(vector_store):
    """Creates the RAG chain for text documents."""
    llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model='llama3-8b-8192')
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question based only on the following context. Provide the answer in Albanian.
        Context: {context}
        Question: {input}
        Answer in Albanian:
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def create_excel_agent(file_path):
    """Creates a Data Analyst Agent for Excel files."""
    llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model='llama3-8b-8192', temperature=0)
    df = pd.read_excel(file_path)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True # We now control this via the UI button
    )
    return agent

# --- STREAMLIT APP ---

st.set_page_config(page_title="BizIntel Scan", layout="wide")
st.title("BizIntel Scan - Analizë Inteligjente e Dokumenteve 🇦🇱")
st.write("Ngarkoni një dokument (PDF, DOCX, TXT, XLSX) dhe bëni pyetje specifike rreth përmbajtjes së tij.")

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

with st.sidebar:
    st.header("1. Ngarko Dokumentin")
    uploaded_file = st.file_uploader("Zgjidhni një dokument", type=["pdf", "docx", "txt", "xlsx"])

    if st.session_state.agent_chain is not None:
        if st.button("Fillo një analizë të re"):
            st.session_state.agent_chain = None
            st.session_state.processed_file = None
            st.rerun()
    
    temp_file_path = None
    if uploaded_file is not None:
        temp_file_path = os.path.join(".", f"temp_{uploaded_file.name}")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if uploaded_file and st.session_state.processed_file != uploaded_file.name:
        st.info(f"Dokumenti i gatshëm për përpunim: `{uploaded_file.name}`")
        
        # --- NEW LOGIC: Differentiate between Excel and other files ---
        is_excel = uploaded_file.name.endswith('.xlsx')

        if is_excel:
            st.warning("Kujdes: Analiza e skedarëve Excel lejon ekzekutimin e kodit për të analizuar të dhënat. Kjo është e sigurt me skedarë të besuar.")
            if st.button("Po, vazhdo me analizën e Excel"):
                with st.spinner("Duke krijuar agjentin e analistit të të dhënave..."):
                    st.session_state.agent_chain = create_excel_agent(temp_file_path)
                    st.session_state.processed_file = uploaded_file.name
                    st.rerun()
        else: # For PDF, DOCX, TXT
            if st.button("Përpuno Dokumentin"):
                with st.spinner("Duke përpunuar dokumentin..."):
                    vector_store = get_vector_store_from_file(temp_file_path)
                    if vector_store is not None:
                        st.session_state.agent_chain = create_rag_chain(vector_store)
                        st.session_state.processed_file = uploaded_file.name
                    st.rerun()

    elif st.session_state.processed_file is not None:
        st.success(f"Dokumenti '{st.session_state.processed_file}' është gati për pyetje.")

    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# Main chat interface
st.header("2. Bëni pyetjen tuaj")

if st.session_state.agent_chain:
    user_question = st.text_input("Shkruani pyetjen tuaj këtu:")
    if user_question:
        with st.spinner("Duke kërkuar përgjigjen..."):
            try:
                # --- NEW LOGIC: The response key is different for agents ---
                is_excel = (
                    st.session_state.processed_file is not None and
                    st.session_state.processed_file.endswith('.xlsx')
                )
                response = st.session_state.agent_chain.invoke({"input": user_question})
                
                st.write("### Përgjigje:")
                if is_excel:
                    st.write(response["output"])
                else: # For RAG chain
                    st.write(response["answer"])
                    with st.expander("Shiko kontekstin e përdorur"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"--- Pjesa e Kontekstit {i+1} ---")
                            st.write(doc.page_content)
            except Exception as e:
                st.error(f"Ndodhi një gabim: {e}")
else:
    st.warning("Ju lutem ngarkoni dhe përpunoni një dokument për të filluar.")