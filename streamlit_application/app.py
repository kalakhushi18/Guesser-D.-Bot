import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor,  create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, json, asyncio, re
from tools import create_summarize_character_tool, create_generate_question_tool, create_trivia_mode
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
    st.subheader("This LLM has multiple Agent Tools")
    st.markdown("1. Guesser Tool")
    st.markdown("2. Summarizer Tool")
    st.markdown("3. Question Asking Tool")

# Load LLM + Embeddings
async def load_llm_and_embeddings():
    llm =   ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.1)
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL_NAME)
    return llm, embeddings

# Load JSON data

@st.cache_resource
def load_combined_vectorstore(_embeddings):
    file_map = [
     "../data/straw_hat_pirates.json",
     "../data/marines.json",
    ]

    index_folder = "faiss_index/all_teams"

    # Reuse existing FAISS index if it exists
    if os.path.exists(index_folder):
        return FAISS.load_local(index_folder, _embeddings, allow_dangerous_deserialization=True)

    # Load all documents from all JSON files
    all_docs = []
    for path in file_map:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, value in data.items():
                text = f"Name: {key}" + "\n".join(f"{k}: {v}" for k, v in value.items())
                all_docs.append(Document(page_content=text, metadata={"character": key}))

                print("All Docs:", all_docs)

    # Split into chunks and index
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(chunks, _embeddings)
    vector_store.save_local(index_folder)
    return vector_store


def extract_core_topic(question: str) -> str:
    # Common starter phrases to strip
    starter_patterns = [
        r"(has|have|is|was|did|does|do|can|could|would)\s+(your\s+)?character\s+(ever\s+)?",
        r"does\s+the\s+character\s+",
        r"is\s+your\s+character\s+"
    ]

    question_clean = question.strip().lower()

    # Strip leading patterns
    for pattern in starter_patterns:
        question_clean = re.sub(pattern, "", question_clean, flags=re.IGNORECASE)

    # Remove trailing punctuation
    question_clean = re.sub(r"[?.!]+$", "", question_clean)

    return question_clean.strip().capitalize()

def extract_simplified_qna(messages):
    pairs = []
    current_question = None

    for msg in messages:
        if isinstance(msg, AIMessage):
            current_question = msg.content.strip()
        elif isinstance(msg, HumanMessage) and current_question:
            simplified = extract_core_topic(current_question)
            pairs.append({
                "trait": simplified,
                "answer": msg.content.strip()
            })
            current_question = None

    return pairs


prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),  ("human", USER), ("placeholder", "{agent_scratchpad}"),])

# Session Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Think of a One Piece character and I will try to guess!")


# Process input
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    llm, embeddings = asyncio.run(load_llm_and_embeddings())
    st.session_state.llm = llm  # for use in tools
    st.session_state.embeddings = embeddings
    
    vector_store = load_combined_vectorstore(embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_input)

    context = "\n\n".join(doc.page_content for doc in docs)

    history  = "\n".join(msg.content for msg in st.session_state.messages)

    combined_input = f"{history}###{context}"

    # Show past chat
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)


    tools = [
        create_generate_question_tool(llm),
        create_summarize_character_tool(llm),
        create_trivia_mode(llm, context)
    ]
    
    
    agent = create_tool_calling_agent(llm, tools, prompt)
   
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.spinner("Summoning the crew's knowledge..."):
        result = agent_executor.invoke({
            "input": combined_input })

    st.session_state.messages.append(AIMessage(content=result["output"]))
    with st.chat_message("assistant"):
        st.markdown(result["output"])
