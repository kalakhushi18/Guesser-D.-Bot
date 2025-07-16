import getpass
import os
# from langchain_core.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import TypedDict, List
import json
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = InMemoryVectorStore(embeddings)

def load_straw_hat_json(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    # print(data)
    for key, value in data.items():
        text = f"Name: {key}\n" + "\n".join(f"{k}: {v}" for k, v in value.items())
        docs.append(Document(page_content=text, metadata={"character": key}))
    return docs
docs = load_straw_hat_json("data\straw_hat_pirates.json")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)



# Index chunks
_ = vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(
    search_type = "similarity",     # "mmr" or "similarity_score_threshold" also work
    search_kwargs = {"k": 4}
)


SYSTEM = """You are an expert assistant.
Answer *only* from the context between <context></context>;
if the answer isn’t there, say “I don't know.”"""
USER = """<context>\n{context}\n</context>\n\nQuestion: {input}"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])

# NEW: wrap llm + prompt in a "stuff-documents" chain → Runnable
combine_docs_chain = create_stuff_documents_chain(llm, prompt)  

# Build the final RAG runnable
rag_chain = create_retrieval_chain(retriever, combine_docs_chain) 

question = "Who is luffy?"
result   = rag_chain.invoke({"input": question})

print(result["answer"])      # grounded answer
print(result["context"])   

