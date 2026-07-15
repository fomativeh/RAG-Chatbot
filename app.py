import os
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import requests

_qp = st.experimental_get_query_params()
if _qp.get("health", [None])[0] == "1":
    st.write("ok")
    st.stop()

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# --- Logging setup ---
logger = logging.getLogger("rag")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# --- Auto-create directories ---
def setup_directories():
    """Create necessary directories on startup"""
    directories = ["data/covid_docs", "chroma_db", "prompts"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directory setup completed")

# Run directory setup
setup_directories()

# Constants
COVID_COLLECTION = "covid_docs"
CUSTOM_COLLECTION_PREFIX = "custom_session_"
COVID_DOCS_DIR = "data/covid_docs"
FIXED_TOP_K = 4

# --- Conversation Memory Functions ---
def get_conversation_history(messages, max_exchanges=3):
    """Get recent conversation history for context"""
    if len(messages) <= 2:  # Need at least 1 exchange (user + assistant)
        return ""
    
    # Get last N exchanges (user + assistant pairs)
    recent_messages = messages[-(max_exchanges * 2):]
    history = "Previous conversation:\n"
    
    for i in range(0, len(recent_messages), 2):
        if i+1 < len(recent_messages):
            user_msg = recent_messages[i]['content']
            assistant_msg = recent_messages[i+1]['content']
            history += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
    
    return history.strip()

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Initialize Chroma client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(name: str):
    client = get_chroma_client()
    embedding_model = get_embedding_model()
    
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    return collection, embedding_model

def delete_session_collection(name: str):
    try:
        client = get_chroma_client()
        client.delete_collection(name)
        logger.info(f"Deleted session collection: {name}")
    except Exception:
        pass

def retrieve_context(query: str, top_k: int, collection_name: str) -> Tuple[List[str], List[str]]:
    try:
        collection, embedding_model = get_or_create_collection(collection_name)
        
        # Check if collection has any documents
        if collection.count() == 0:
            logger.warning(f"Collection {collection_name} is empty")
            return [], []
        
        query_embedding = embedding_model.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas"]
        )
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        # Create citation labels
        labels = []
        for i, meta in enumerate(metas):
            source = meta.get('source', 'Unknown')
            page = meta.get('page', meta.get('page_number', 'N/A'))
            labels.append(f"{source} (Page {page})")
        
        logger.info(f"Retrieved {len(docs)} context chunks for query: {query}")
        return docs, labels
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return [], []

@st.cache_data(show_spinner=False)
def ingest_covid_documents() -> bool:
    """Ingest COVID documents on first run - cached to run only once"""
    if not os.path.exists(COVID_DOCS_DIR):
        os.makedirs(COVID_DOCS_DIR, exist_ok=True)
        return True
    
    # Check if collection already exists and has content
    try:
        collection, _ = get_or_create_collection(COVID_COLLECTION)
        doc_count = collection.count()
        if doc_count > 0:
            logger.info(f"COVID documents already ingested ({doc_count} chunks)")
            return True
        else:
            logger.info("COVID collection exists but is empty, will attempt ingestion")
    except Exception as e:
        logger.info(f"COVID collection doesn't exist yet, will create: {e}")
    
    # Look for PDF files in COVID docs directory
    pdf_files = [f for f in os.listdir(COVID_DOCS_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning("No PDF files found in COVID docs directory")
        return False
    
    all_texts = []
    all_metas = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(COVID_DOCS_DIR, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            for doc in documents:
                chunks = splitter.split_documents([doc])
                for chunk in chunks:
                    all_texts.append(chunk.page_content)
                    metadata = {
                        'source': pdf_file,
                        'page': chunk.metadata.get('page', 0) + 1,
                        'chunk_index': len(all_texts) - 1
                    }
                    all_metas.append(metadata)
            
            logger.info(f"Successfully processed {pdf_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            continue
    
    if all_texts:
        try:
            collection, embedding_model = get_or_create_collection(COVID_COLLECTION)
            embeddings = embedding_model.embed_documents(all_texts)
            ids = [f"doc_{i}" for i in range(len(all_texts))]
            
            collection.add(
                embeddings=embeddings,
                documents=all_texts,
                metadatas=all_metas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {len(all_texts)} chunks from COVID documents")
            return True
        except Exception as e:
            logger.error(f"Error ingesting COVID documents: {e}")
            return False
    
    return False

def ingest_uploaded_files(uploaded_files) -> str:
    """Ingest uploaded files and return session collection name"""
    session_id = st.session_state.get('session_id', 'default_session')
    collection_name = f"{CUSTOM_COLLECTION_PREFIX}{session_id}"
    
    all_texts = []
    all_metas = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                temp_path = f".temp_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                
                for doc in documents:
                    chunks = splitter.split_documents([doc])
                    for chunk in chunks:
                        all_texts.append(chunk.page_content)
                        metadata = {
                            'source': file_name,
                            'page': chunk.metadata.get('page', 0) + 1,
                            'chunk_index': len(all_texts) - 1
                        }
                        all_metas.append(metadata)
                
                os.remove(temp_path)
                
            elif file_extension in ['txt', 'md']:
                content = uploaded_file.getvalue().decode('utf-8')
                chunks = splitter.split_documents([LCDocument(page_content=content)])
                
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk.page_content)
                    metadata = {
                        'source': file_name,
                        'page': 1,
                        'chunk_index': i
                    }
                    all_metas.append(metadata)
            
            logger.info(f"Processed {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            continue
    
    if all_texts:
        try:
            collection, embedding_model = get_or_create_collection(collection_name)
            embeddings = embedding_model.embed_documents(all_texts)
            ids = [f"upload_{i}" for i in range(len(all_texts))]
            
            collection.add(
                embeddings=embeddings,
                documents=all_texts,
                metadatas=all_metas,
                ids=ids
            )
            
            logger.info(f"Ingested {len(all_texts)} chunks from uploaded files")
            return collection_name
        except Exception as e:
            logger.error(f"Error ingesting uploaded files: {e}")
            return None
    
    return None

def load_prompt(mode: str) -> str:
    """Load prompt based on mode"""
    prompt_file = f"prompts/{'covid_assistant' if mode == 'covid' else 'custom_bot'}_prompt.txt"
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        logger.info(f"Loaded prompt from {prompt_file}")
        return prompt_content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        return "Context: {context}\n\nQuestion: {question}\n\nAnswer:"

def format_context(documents: List[str], labels: List[str]) -> str:
    """Format context documents with citations"""
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, (doc, label) in enumerate(zip(documents, labels)):
        page_num = "N/A"
        if "Page " in label:
            try:
                page_num = label.split("Page ")[1].split(")")[0]
            except:
                page_num = "N/A"
        
        context_parts.append(f"[Page {page_num}]:\n{doc}\n")
    return "\n".join(context_parts)

def get_secret(key: str):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key)

def openrouter_direct_call(prompt: str) -> str:
    """Direct API call to OpenRouter as fallback"""
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not configured"
    
    base_url = "https://openrouter.ai/api/v1"
    model = "openai/gpt-3.5-turbo"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenRouter direct API call failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

def generate_llm_response(prompt: str) -> str:
    openrouter_key = get_secret("OPENROUTER_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    if openrouter_key:
        try:
            llm = ChatOpenAI(
                model="openai/gpt-3.5-turbo",
                temperature=0.2,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                max_retries=2,
                request_timeout=30
            )
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")

    if openai_key:
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                api_key=openai_key,
                base_url="https://api.openai.com/v1",
                max_retries=2,
                request_timeout=30
            )
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            logger.error(f"OpenAI fallback error: {e}")

    if openrouter_key:
        return openrouter_direct_call(prompt)

    return "Error: No LLM API key configured (set OPENROUTER_API_KEY or OPENAI_API_KEY in Streamlit secrets or .env)."

def main():
    st.set_page_config(
        page_title="Dual Mode RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(hash(str(os.urandom(16))))
    
    if 'messages_by_mode' not in st.session_state:
        st.session_state.messages_by_mode = {
            'covid': [],
            'custom': []
        }
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'covid'
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'custom_collection' not in st.session_state:
        st.session_state.custom_collection = None
    
    if 'covid_ingested' not in st.session_state:
        st.session_state.covid_ingested = False
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Chatbot Mode")
        
        mode = st.radio(
            "Select Chatbot Mode:",
            options=["🩺 COVID-19 Assistant", "🧠 Your Custom Bot"],
            index=0 if st.session_state.current_mode == 'covid' else 1
        )
        
        if "COVID-19" in mode:
            st.session_state.current_mode = 'covid'
            try:
                collection, _ = get_or_create_collection(COVID_COLLECTION)
                doc_count = collection.count()
                if doc_count > 0:
                    st.session_state.covid_ingested = True
                    st.info(f"✅ COVID guidelines ready ({doc_count} chunks)")
                else:
                    st.session_state.covid_ingested = False
            except:
                st.session_state.covid_ingested = False
            
            if not st.session_state.covid_ingested:
                with st.spinner("Loading COVID guidelines..."):
                    if ingest_covid_documents():
                        st.session_state.covid_ingested = True
                        try:
                            collection, _ = get_or_create_collection(COVID_COLLECTION)
                            doc_count = collection.count()
                            if doc_count > 0:
                                st.success(f"✅ COVID guidelines loaded! ({doc_count} chunks)")
                            else:
                                st.warning("⚠️ No COVID documents found. Add PDF files to data/covid_docs/")
                        except:
                            st.error("❌ Error accessing COVID collection")
                    else:
                        st.error("❌ Failed to load COVID guidelines")
        else:
            st.session_state.current_mode = 'custom'
            
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'md'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or MD files"
            )
            
            if uploaded_files and uploaded_files != st.session_state.uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                with st.spinner("Processing uploaded files..."):
                    collection_name = ingest_uploaded_files(uploaded_files)
                    if collection_name:
                        st.session_state.custom_collection = collection_name
                        st.success(f"Processed {len(uploaded_files)} files!")
                        st.session_state.messages_by_mode['custom'] = []
                    else:
                        st.error("Failed to process uploaded files")
            
            if st.session_state.uploaded_files:
                st.subheader("Uploaded Files")
                for file in st.session_state.uploaded_files:
                    st.write(f"📄 {file.name}")
        
        # Clear conversation button
        st.markdown("---")
        if st.button("🧹 Clear Conversation"):
            current_mode = st.session_state.current_mode
            st.session_state.messages_by_mode[current_mode] = []
            st.success("Conversation cleared!")
            st.rerun()
    
    # Main content area
    if st.session_state.current_mode == 'covid':
        st.title("🩺 COVID-19 Treatment Assistant")
        covid_doc_url = os.getenv('COVID_DOC_URL')
        st.caption(f'I respond based on the [NIH COVID-19 Treatment Guidelines]({covid_doc_url})')
    else:
        st.title("🧠 Your RAG Chatbot")
        st.caption("I am an assistant trained to respond based on the documents you uploaded")
    
    # Display chat messages
    current_messages = st.session_state.messages_by_mode[st.session_state.current_mode]
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if we can chat
    can_chat = True
    error_message = ""
    
    if st.session_state.current_mode == 'covid':
        try:
            collection, _ = get_or_create_collection(COVID_COLLECTION)
            if collection.count() == 0:
                can_chat = False
                error_message = "No COVID-19 documents found. Please add PDF files to the data/covid_docs/ directory."
        except:
            can_chat = False
            error_message = "Error accessing COVID-19 document collection."
    else:
        if not st.session_state.custom_collection:
            can_chat = False
            error_message = "Please upload documents first to start chatting..."
    
    # Chat input
    placeholder = {
        'covid': "Ask a question about COVID-19 treatment guidelines...",
        'custom': "Ask a question about your uploaded documents..."
    }[st.session_state.current_mode]
    
    if not can_chat:
        st.warning(error_message)
        placeholder = error_message
    
    if prompt := st.chat_input(placeholder=placeholder, disabled=not can_chat):
        # Add user message to chat
        current_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Determine which collection to use
        collection_name = (
            COVID_COLLECTION if st.session_state.current_mode == 'covid' 
            else st.session_state.custom_collection
        )
        
        if not collection_name:
            response = "Please upload documents first to enable the custom chatbot."
        else:
            try:
                # Retrieve context
                with st.spinner("🔍 Searching documents..."):
                    docs, labels = retrieve_context(prompt, FIXED_TOP_K, collection_name)
                
                if not docs:
                    if st.session_state.current_mode == 'covid':
                        response = "I don't have information about that topic."
                    else:
                        response = "I couldn't find relevant information in the uploaded documents to answer your question."
                else:
                    # Prepare context and conversation history
                    context_block = format_context(docs, labels)
                    conversation_history = get_conversation_history(current_messages)
                    
                    # Load and format prompt
                    system_prompt = load_prompt(st.session_state.current_mode)
                    
                    if "{conversation_history}" in system_prompt and "{context}" in system_prompt and "{question}" in system_prompt:
                        final_prompt = system_prompt.format(
                            conversation_history=conversation_history,
                            context=context_block,
                            question=prompt
                        )
                    else:
                        # Fallback if prompt doesn't have all placeholders
                        final_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_history}\n\nContext:\n{context_block}\n\nQuestion: {prompt}\n\nAnswer:"
                    
                    # Generate response
                    with st.spinner("💭 Generating response..."):
                        response = generate_llm_response(final_prompt)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                error_msg = st.empty()
                error_msg.error("⚠️ Sorry, I encountered an error. Please try again in a moment.")
                import time
                time.sleep(3)
                error_msg.empty()
                response = "I apologize, but I'm having trouble generating a response right now. Please try again."
        
        # Add assistant response to chat
        current_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.rerun()

if __name__ == "__main__":
    main()
