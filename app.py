import streamlit as st
import json
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
import google.generativeai as genai
import asyncio
import sys
from langchain.memory import ConversationBufferMemory


load_dotenv()


if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

FAISS_INDEX_PATH = ""
EMBEDDING_MODEL_NAME = ""


# App configuration
st.set_page_config(page_title="Guesser D. Bot", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
        }
        .title {
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 48px;
            color: #FFD700;
            text-shadow: 2px 2px #8B0000;
            text-align: center;
            margin-bottom: 0;
        }
        .subtitle {
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 20px;
            text-align: center;
            margin-top: 0;
            color: #333333;
        }
        .chatbox {
            background-color: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
        .footer {
            font-size: 12px;
            text-align: center;
            color: gray;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Guesser D. Bot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>An interactive chatbot that tries to guess your favorite Character!</div>", unsafe_allow_html=True)
st.markdown("---")


# Sidebar info
with st.sidebar:
    st.image("one-peice.jpg", caption="Straw Hat Pirates")
    st.subheader("About Guesser D. Bot")
    st.markdown("""
    <div style="font-size: 15px; line-height: 1.6;">
        🤖 Powered by <b>AI + LangChain</b><br>
        🧠 Uses <b>FAISS vector search</b> for memory<br>
        🌐 Supports <b>Gemini & Ollama</b> models<br>
        📚 Uses <b>Google + Local Embeddings</b><br>
        🕵️‍♂️ Inspired by <b>Akinator</b>, built for <b>One Piece</b> fans!
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Choose your LLM")
    selected_model = st.selectbox("Select a language model", ["Gemini", "Ollama"])
    st.markdown(f"✅ **Model selected:** {selected_model}")

    st.markdown("---")
    st.subheader("Choose your Team")
    selected_team = st.selectbox("Which crew do you want to play with?", [
        "Straw Hat Pirates",
        "Navy",
        "Emperors"
    ])
    st.markdown(f"🏴‍☠️ **Playing as:** {selected_team}")

# Load appropriate LLM + embeddings
async def load_llm_and_embeddings(model_choice):
    if model_choice == "Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        llm = ChatOllama(model="llama2:latest", temperature=0.3)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embeddings

# Load JSON for selected team
@st.cache_resource
def load_vectorstore_from_team(team: str, _embeddings):
    file_map = {
        "Straw Hat Pirates": "data/straw_hat_pirates.json",
        "Navy": "data/navy.json",
        "Emperors": "data/emperors.json"
    }
    path = file_map.get(team)
    if not path:
        st.error("Team data file not found.")
        st.stop()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for key, value in data.items():
        text = f"Name: {key}\n" + "\n".join(f"{k}: {v}" for k, v in value.items())
        docs.append(Document(page_content=text, metadata={"character": key}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, _embeddings)

# Set up prompt and chain
SYSTEM = """You are an expert assistant.
Answer *only* from the context between <context></context>;
if the answer isn’t there, say “I don't know.”"""
USER = """<context>\n{context}\n</context>\n\nQuestion: {input}"""


prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])

# Chat session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show full chat history before new input
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask me anything about One Piece teams...")


if query:
    llm, embeddings = asyncio.run(load_llm_and_embeddings(selected_model))
    vector_store = load_vectorstore_from_team(selected_team, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # RAG pipeline
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Summoning your crew's knowledge..."):
        result = rag_chain.invoke({"input": query})
        response = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


# Footer
st.markdown("<div class='footer'>Made with ❤️ by a One Piece fan. Powered by Streamlit + LangChain</div>", unsafe_allow_html=True)