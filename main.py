
import streamlit as st
import os
import io
import shutil
import base64
from typing import Optional

from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- KittenTTS imports ---
from kittentts import KittenTTS
import soundfile as sf

# --- components for autoplay ---
import streamlit.components.v1 as components

# ================================
# Configuration
# ================================
st.set_page_config(page_title="Conversational RAG with Ollama", layout="wide")
st.title("Conversational RAG with Ollama, Web & Local Docs 📚")

SOURCE_URLS = []
LOCAL_DOCS_DIR = "./local_docs"
OLLAMA_LLM = "sea_turtle_llama3_1_8b_q4_k_m"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_DIR = "./chroma_db_ollama_memory_st"

DEFAULT_KITTEN_MODEL_ID = "KittenML/kitten-tts-nano-0.1"
DEFAULT_SAMPLE_RATE = 24000
USE_TTS_GPU = False

# Voices from KittenTTS README
AVAILABLE_VOICES = [
    "expr-voice-2-m", "expr-voice-2-f",
    "expr-voice-3-m", "expr-voice-3-f",
    "expr-voice-4-m", "expr-voice-4-f",
    "expr-voice-5-m", "expr-voice-5-f",
]

# ================================
# Helpers: TTS (KittenTTS)
# ================================
@st.cache_resource(show_spinner="Loading KittenTTS (first run may download weights)...")
def get_tts(model_id: str = DEFAULT_KITTEN_MODEL_ID, use_gpu: bool = USE_TTS_GPU):
    return KittenTTS(model_id)

def synthesize_to_wav_bytes(text: str, tts: KittenTTS, voice: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    if not text or not text.strip():
        return b""
    wav = tts.generate(text + '....', voice=voice)
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    buf.seek(0)
    return buf.getvalue()

def autoplay_audio(audio_bytes: bytes, show_controls: bool = False):
    """
    Inject a hidden <audio> element that autoplays in the browser.
    No 'key' argument is used to maintain compatibility with older Streamlit versions.
    """
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    controls_attr = "controls" if show_controls else ""
    html = f"""
        <audio id="tts-audio" {controls_attr} autoplay style="display:none;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        <script>
            const el = document.getElementById('tts-audio');
            if (el) {{
                el.play().catch(() => {{
                    el.style.display = 'block';
                    el.setAttribute('controls','');
                    el.play().catch(() => {{ }});
                }});
            }}
        </script>
    """
    components.html(html, height=0, scrolling=False)

# ================================
# RAG Setup Function (Cached)
# ================================
@st.cache_resource(show_spinner="Setting up the RAG chain... Please wait.")
def setup_rag_chain():
    all_documents = []

    # Load from local directory
    with st.status(f"Loading documents from local directory: {LOCAL_DOCS_DIR}...", expanded=True) as status:
        if os.path.exists(LOCAL_DOCS_DIR):
            try:
                loader = DirectoryLoader(LOCAL_DOCS_DIR, glob="*.txt", loader_cls=TextLoader, show_progress=True)
                local_docs = loader.load()
                if local_docs:
                    st.write(f"✓ Loaded {len(local_docs)} documents from {LOCAL_DOCS_DIR}")
                    all_documents.extend(local_docs)
                else:
                    st.write(f"✗ No .txt documents found in {LOCAL_DOCS_DIR}")
            except Exception as e:
                st.error(f"Error loading documents from local directory: {e}")
        else:
            st.warning(f"Local documents directory not found: {LOCAL_DOCS_DIR}. Skipping.")
        status.update(label="Local loading complete!", state="complete")

    if not all_documents:
        st.error("No documents were loaded. The application cannot proceed. Please check your sources.")
        st.stop()

    # Split, embed, and create vector store
    with st.status("Splitting documents, embedding, and creating vector store...", expanded=True) as status:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(all_documents)
        st.write(f"Split into {len(split_documents)} chunks.")

        try:
            embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
            st.write("Ollama Embeddings initialized.")
        except Exception as e:
            st.error(f"Error initializing Ollama Embeddings. Make sure Ollama is running and model '{OLLAMA_EMBEDDING_MODEL}' is pulled.")
            st.error(e)
            st.stop()

        if os.path.exists(CHROMA_DB_DIR):
            st.write(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
            try:
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
            except Exception as e:
                st.warning(f"Error loading Chroma DB: {e}. Rebuilding.")
                shutil.rmtree(CHROMA_DB_DIR)
                vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
        else:
            st.write(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
            vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)

        st.write("Vector store is ready.")
        status.update(label="Vector store created successfully!", state="complete")

    # Setup RAG chain
    with st.status("Initializing Ollama LLM and RAG chain...", expanded=True) as status:
        try:
            ollama_llm = ChatOllama(model=OLLAMA_LLM, temperature=0.1)
            st.write(f"Ollama LLM initialized with model: {OLLAMA_LLM}")
        except Exception as e:
            st.error(f"Error initializing Ollama LLM. Make sure Ollama server is running and model '{OLLAMA_LLM}' is pulled.")
            st.error(e)
            st.stop()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Return only the query and nothing else."),
        ])

        history_aware_retriever_chain = create_history_aware_retriever(
            ollama_llm, retriever, history_aware_retriever_prompt
        )

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Short answer with not more than 30 words. Answer the user's question based ONLY on the below context.
If you cannot find the answer in the provided context, state that you do not have enough information from the provided sources to answer. Do not make up an answer.

Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        stuff_documents_chain = create_stuff_documents_chain(ollama_llm, rag_prompt)
        conversational_rag_chain = create_retrieval_chain(history_aware_retriever_chain, stuff_documents_chain)
        st.write("Conversational RAG chain is ready.")
        status.update(label="RAG chain setup complete!", state="complete")

    return conversational_rag_chain

# ================================
# Main Application Logic
# ================================

try:
    rag_chain = setup_rag_chain()
except Exception as e:
    st.error("An error occurred during the setup of the RAG chain.")
    st.error(e)
    st.stop()

# --- Session state ---
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "last_answer_text" not in st.session_state:
    st.session_state.last_answer_text = ""
if "last_answer_audio_bytes" not in st.session_state:
    st.session_state.last_answer_audio_bytes = b""
if "selected_voice" not in st.session_state:
    # default to the first female voice
    st.session_state.selected_voice = "expr-voice-4-f"

# Sidebar controls
with st.sidebar:
    st.subheader("Text-to-Speech (KittenTTS)")
    auto_speak = st.checkbox("Auto-speak responses", value=True, help="Speak automatically after each model response.")
    selected_voice = st.selectbox("Voice", AVAILABLE_VOICES, index=AVAILABLE_VOICES.index(st.session_state.selected_voice))
    st.session_state.selected_voice = selected_voice
    st.caption("Voices are defined by the KittenTTS nano model.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am a RAG assistant. How can I help you today?")
    ]

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

prompt = st.chat_input("Ask a question about the documents...")
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state.is_generating = True
    try:
        with st.spinner("Thinking..."):
            chat_history_for_chain = [msg for msg in st.session_state.messages[:-1]]
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history_for_chain
            })
            answer = response.get('answer', "Sorry, I could not generate an answer.")
            st.session_state.messages.append(AIMessage(content=answer))
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)

            st.session_state.last_answer_text = answer
            st.session_state.last_answer_audio_bytes = b""

            # --- Auto-speak using the selected voice ---
            if auto_speak and answer.strip():
                with st.spinner(f"Synthesizing speech ({st.session_state.selected_voice})..."):
                    try:
                        tts = get_tts()
                        audio_bytes = synthesize_to_wav_bytes(
                            answer, tts, voice=st.session_state.selected_voice, sample_rate=DEFAULT_SAMPLE_RATE
                        )
                        st.session_state.last_answer_audio_bytes = audio_bytes
                        autoplay_audio(audio_bytes, show_controls=False)
                    except Exception as e:
                        st.error(f"TTS synthesis failed: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.append(AIMessage(content=f"Sorry, I encountered an error: {e}"))
    finally:
        st.session_state.is_generating = False

# Visible player as fallback / replay
if st.session_state.last_answer_audio_bytes:
    st.audio(st.session_state.last_answer_audio_bytes, format="audio/wav")
    st.caption("If autoplay is blocked, press play above.")
