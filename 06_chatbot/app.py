"""
Module 6: Complete LangChain Chatbot with Gradio UI
=====================================================
This is the final project — a working chatbot that combines:
  ✅ LCEL chains (prompt | model | parser)
  ✅ Memory (RunnableWithMessageHistory)
  ✅ RAG (document retrieval with FAISS)
  ✅ Gradio web interface

HOW TO RUN:
  python app.py
  Then open: http://localhost:7860

SETUP:
  pip install gradio langchain-huggingface faiss-cpu sentence-transformers
"""

import os
import gradio as gr
from dotenv import load_dotenv
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ===========================================================
# 1. KNOWLEDGE BASE — Add your own docs here!
# ===========================================================

KNOWLEDGE_BASE = [
    Document(page_content="LangChain is a framework for building LLM-powered applications.",
             metadata={"source": "intro"}),
    Document(page_content="LCEL uses the pipe (|) operator to chain components: prompt | model | parser.",
             metadata={"source": "lcel"}),
    Document(page_content="Memory systems store conversation history so the AI can recall past turns.",
             metadata={"source": "memory"}),
    Document(page_content="RAG retrieves relevant document chunks and injects them as context before generating.",
             metadata={"source": "rag"}),
    Document(page_content="RunnableParallel executes multiple chains concurrently, reducing latency.",
             metadata={"source": "parallel"}),
    Document(page_content="RunnableLambda wraps any Python function as a chain step for custom logic.",
             metadata={"source": "lambda"}),
    Document(page_content="Agents use LLMs as reasoning engines to decide which tools to call.",
             metadata={"source": "agents"}),
    Document(page_content="HuggingFace embedding models like all-MiniLM-L6-v2 run locally, no API key needed.",
             metadata={"source": "embeddings"}),
]

# ===========================================================
# 2. INITIALISE COMPONENTS
# ===========================================================

print("🔧 Initializing components...")

# LLM
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Embeddings (free, runs locally)
print("  Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Vector store
print("  Building vector store...")
splitter  = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(KNOWLEDGE_BASE)
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# Memory store
session_store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# ===========================================================
# 3. RAG + MEMORY CHAIN
# ===========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_context(input_dict):
    question = input_dict.get("input", "")
    docs     = retriever.invoke(question)
    return format_docs(docs)

# Prompt: history + RAG context + new question
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful LangChain tutor. Use the context below when relevant.\n"
     "Always be concise and clear.\n\n"
     "Context from knowledge base:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Build chain: retrieve context → inject into prompt → model → parser
chain = (
    RunnablePassthrough.assign(context=retrieve_context)
    | prompt
    | model
    | StrOutputParser()
)

# Wrap with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("✅ Chatbot ready!\n")

# ===========================================================
# 4. GRADIO INTERFACE
# ===========================================================

def chat(message: str, history: list, session_id: str) -> str:
    """Handle one chat turn."""
    if not message.strip():
        return ""
    config = {"configurable": {"session_id": session_id}}
    try:
        response = chain_with_memory.invoke({"input": message}, config)
        return response
    except Exception as e:
        return f"❌ Error: {e}\n\nMake sure your OPENAI_API_KEY is set in .env"

def clear_memory(session_id: str) -> str:
    """Clear conversation memory for this session."""
    if session_id in session_store:
        del session_store[session_id]
    return "Memory cleared! Starting fresh."

# Build Gradio UI
with gr.Blocks(title="LangChain Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦜 LangChain Chatbot
    A complete LangChain demo with **LCEL chains**, **conversation memory**, and **RAG**.

    **Try these prompts:**
    - "What is LCEL and how does it work?"
    - "My name is Alice. What are LangChain memory types?"
    - "What's my name?" *(tests memory)*
    - "How does RAG work?"
    """)

    session_id = gr.State(value="default-user")

    chatbot = gr.Chatbot(height=450, label="Conversation")

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask anything about LangChain...",
            label="Your message",
            scale=4,
        )
        send_btn = gr.Button("Send 🚀", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("🗑  Clear Memory")
        status    = gr.Textbox(label="Status", interactive=False, scale=3)

    # Wire up events
    def respond(message, history, sid):
        reply   = chat(message, history, sid)
        history = history + [[message, reply]]
        return history, ""

    send_btn.click(respond, [msg_box, chatbot, session_id], [chatbot, msg_box])
    msg_box.submit(respond, [msg_box, chatbot, session_id], [chatbot, msg_box])
    clear_btn.click(lambda sid: ([], clear_memory(sid)), [session_id], [chatbot, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
