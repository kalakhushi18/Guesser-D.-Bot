import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, json, asyncio
from tools import create_summarize_character_tool, create_generate_question_tool, create_a_guess_heatmap, create_emotional_bot_tool
from prompts import SYSTEM_PROMPT, USER

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GEMINI_EMBEDDING_MODEL_NAME = "models/embedding-001"
GEMINI_LLM_MODEL = "gemini-2.5-flash"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama2:latest"

# Streamlit Page Config
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

# --- Sidebar --- #
with st.sidebar:
    st.image("one-peice.jpg", caption="One Piece By Eiichiro Oda")
    st.markdown("---")
    st.subheader("Choose your LLM")
    selected_model = st.selectbox("Select a language model", ["Gemini", "Ollama"])  #Selecting which model to use
    st.markdown(f"✅ **Model selected:** {selected_model}")

    st.markdown("---")
    st.subheader("This LLM has multiple Agent Tools")
    st.markdown("1. Guesser Tool")
    st.markdown("2. Summarizer Tool")
    st.markdown("3. Question Asking Tool")
   
    # selected_team = st.selectbox("Which crew do you want to play with?", [
    #     "Straw Hat Pirates",
    #     "Marines",
    #     "Emperors",
    #     "General"
    # ]) #Selecting which team to play for
    # st.markdown(f"🏴‍☠️ **Playing as:** {selected_team}")

# Load LLM + Embeddings
async def load_llm_and_embeddings(model_choice):
    if model_choice == "Gemini":
        llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.2)
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL_NAME, task_type="RETRIEVAL_DOCUMENT")
    else:
        llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    return llm, embeddings

# Load JSON data

@st.cache_resource
def load_combined_vectorstore(_embeddings):
    file_map = {
        "Straw Hat Pirates": "data/straw_hat_pirates.json",
        "Marines": "data/marines.json",
        "seven_warlords":"data/seven_warlords.json",
    }   

    index_folder = "faiss_index/all_teams"

    # Reuse existing FAISS index if it exists
    if os.path.exists(index_folder):
        return FAISS.load_local(index_folder, _embeddings, allow_dangerous_deserialization=True)

    # Load all documents from all JSON files
    all_docs = []
    for team, path in file_map.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, value in data.items():
                text = f"Name: {key}\nTeam: {team}\n" + "\n".join(f"{k}: {v}" for k, v in value.items())
                all_docs.append(Document(page_content=text, metadata={"team": team, "character": key}))

    # Split into chunks and index
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(chunks, _embeddings)

    vector_store.save_local(index_folder)
    return vector_store


prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),  ("human", USER), ("placeholder", "{agent_scratchpad}"),])

# Session Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Think of a One Piece character and I will try to guess!")


# Process input
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    llm, embeddings = asyncio.run(load_llm_and_embeddings(selected_model))
    st.session_state.llm = llm  # for use in tools
    st.session_state.embeddings = embeddings
    
    vector_store = load_combined_vectorstore(embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_input)

    context = "\n\n".join(doc.page_content for doc in docs)

    history  = "\n".join(msg.content for msg in st.session_state.messages)

    combined_input = f"{history}####{context}"

    # Show past chat
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

    tools = [
        create_generate_question_tool(llm),
        create_summarize_character_tool(llm),
    ]

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.spinner("Summoning the crew's knowledge..."):
        result = agent_executor.invoke({
            "input": combined_input })

    st.session_state.messages.append(AIMessage(content=result["output"]))
    with st.chat_message("assistant"):
        st.markdown(result["output"])
