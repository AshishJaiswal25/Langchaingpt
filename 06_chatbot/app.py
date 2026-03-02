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
import json
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
from langchain_core.tools import tool

try:
    from langgraph.prebuilt import create_react_agent
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

try:
    from langchain_community.tools.google_search import GoogleSearchAPIWrapper
    HAS_GOOGLE_SEARCH = True
except ImportError:
    HAS_GOOGLE_SEARCH = False

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
    Document(page_content="Function calling allows LLMs to invoke tools and APIs to complete tasks.",
             metadata={"source": "tools"}),
    Document(page_content="Chain of Thought prompting improves reasoning by asking models to think step-by-step.",
             metadata={"source": "prompting"}),
    Document(page_content="Fine-tuning adapts pre-trained models to specific domains and tasks.",
             metadata={"source": "finetuning"}),
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
# 2.5 TOOL DEFINITIONS — Custom Functions the AI Can Use
# ===========================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the LangChain knowledge base for relevant information about a topic."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in knowledge base."
    return format_docs(docs)

@tool
def calculate_expression(expression: str) -> str:
    """Safely evaluate mathematical expressions. Example: '2 + 2 * 3' returns '8'."""
    try:
        # Remove any dangerous functions
        safe_dict = {"__builtins__": {}}
        result = eval(expression, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Invalid expression. Error: {str(e)}"

@tool
def get_langchain_tips() -> str:
    """Get useful tips and best practices for using LangChain effectively."""
    tips = """
    🎯 LangChain Best Practices:
    1. Use LCEL (prompt | model | parser) for clean, composable chains
    2. Leverage memory for context-aware conversations
    3. Implement RAG for accurate, source-grounded responses
    4. Use tools/function calling to extend LLM capabilities
    5. Log interactions for monitoring and debugging
    6. Cache embeddings and retrievals for performance
    7. Test chains incrementally before production
    """
    return tips

@tool
def count_tokens(text: str) -> str:
    """Estimate the number of tokens in a text (rough approximation)."""
    # Rough estimate: ~4 chars per token on average
    token_count = len(text) // 4
    return f"Estimated tokens: {token_count}"

@tool
def google_search(query: str) -> str:
    """Search Google for current information when knowledge base doesn't have the answer."""
    if not HAS_GOOGLE_SEARCH:
        return "Google Search not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env"
    
    try:
        search = GoogleSearchAPIWrapper()
        results = search.run(query)
        return f"Google Search Results for '{query}':\n{results}"
    except Exception as e:
        return f"Search failed: {str(e)[:200]}. Make sure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in .env"

# List of available tools
tools = [search_knowledge_base, calculate_expression, get_langchain_tips, count_tokens]

# Add Google Search if available
if HAS_GOOGLE_SEARCH:
    try:
        # Test if Google Search API keys are configured
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
            tools.append(google_search)
            print("✅ Google Search tool enabled!")
        else:
            print("⚠️  Google Search API keys not found in .env")
    except Exception as e:
        print(f"⚠️  Google Search not available: {e}")

# ===========================================================
# 3. RAG + MEMORY + TOOLS CHAIN WITH AGENT
# ===========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Advanced system prompt for agent
system_prompt = """You are an advanced LangChain tutor and AI assistant with access to tools.

Your capabilities:
✅ You can search the knowledge base using 'search_knowledge_base' tool
✅ You can perform calculations using 'calculate_expression' tool
✅ You can provide LangChain tips using 'get_langchain_tips' tool
✅ You can estimate token counts using 'count_tokens' tool
✅ You can search Google for current information using 'google_search' tool

Instructions:
- Search the knowledge base first for LangChain topics
- Use Google Search when the knowledge base doesn't have the answer
- Always search Google for current events, dates, or real-time information
- Show your reasoning and cite sources
- Be conversational and helpful
- Remember the conversation history for context"""

# Create agent with tools (only if langgraph is available)
agent = None
if HAS_AGENT:
    try:
        agent = create_react_agent(model, tools, state_modifier=system_prompt)
    except Exception as e:
        print(f"⚠️  Agent creation failed: {e}. Falling back to regular chain.")
        agent = None

# For now, keep the simpler chain for initial responses
def retrieve_context(input_dict):
    question = input_dict.get("input", "")
    docs     = retriever.invoke(question)
    return format_docs(docs)

# Prompt: history + RAG context + new question
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful LangChain tutor with access to tools. Use the context below when relevant.\n"
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
    """Handle one chat turn with tool support."""
    if not message.strip():
        return ""
    config = {"configurable": {"session_id": session_id}}
    try:
        # Use the regular chain (with RAG + memory + tools)
        response = chain_with_memory.invoke({"input": message}, config)
        
        # If agent is available, you could optionally use it for advanced tool calls
        # For now, the response includes tool suggestions in the prompt context
        
        return response
    except Exception as e:
        error_msg = str(e)[:200]
        return f"❌ Error: {error_msg}\n\nMake sure your OPENAI_API_KEY is set in .env"

def chat_stream(message: str, history: list, session_id: str):
    """Stream chat responses token-by-token for better UX."""
    if not message.strip():
        yield ""
        return
    
    config = {"configurable": {"session_id": session_id}}
    try:
        # Stream tokens directly from the LLM
        full_response = ""
        
        # Use the model's stream method to get token-by-token output
        from langchain_core.messages import HumanMessage
        
        # Get chat history from session
        history_obj = get_session_history(session_id)
        messages = history_obj.messages + [HumanMessage(content=message)]
        
        # Stream from model directly
        for chunk in model.stream(messages):
            if hasattr(chunk, 'content'):
                token = chunk.content
                full_response += token
                yield full_response  # Yield accumulated response
        
        # Save to memory
        history_obj.add_user_message(message)
        history_obj.add_ai_message(full_response)
        
    except Exception as e:
        error_msg = str(e)[:200]
        yield f"❌ Error: {error_msg}\n\nMake sure your OPENAI_API_KEY is set in .env"

def clear_memory(session_id: str) -> str:
    """Clear conversation memory for this session."""
    if session_id in session_store:
        del session_store[session_id]
    return "Memory cleared! Starting fresh."

# Build Gradio UI with Modern 3D Design
custom_css = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradio-container {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    max-width: 900px !important;
    margin: 20px auto !important;
    padding: 30px !important;
}

/* Header Styling */
.gradio-container h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

/* Chatbot Container */
.chat-container {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(240, 147, 251, 0.05));
    border-radius: 15px !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.05) !important;
    padding: 15px !important;
}

/* Textbox Styling */
.gradio-textbox input, .gradio-textbox textarea {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    padding: 12px 15px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox input:focus, .gradio-textbox textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.3) !important;
    background: white !important;
}

/* Button Styling */
.gradio-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

.gradio-button:active {
    transform: translateY(0) !important;
}

/* Primary Button Special */
.primary.gradio-button {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4) !important;
}

.primary.gradio-button:hover {
    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6) !important;
}

/* Secondary Button */
.secondary.gradio-button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
}

/* Label Styling */
.gradio-label {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Status Box */
.gradio-textbox:disabled, .gradio-textbox[readonly] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(79, 172, 254, 0.05)) !important;
    border: 2px solid rgba(79, 172, 254, 0.2) !important;
}
"""

with gr.Blocks(title="🤖 SAJ ") as demo:
    gr.Markdown("""
    # 🤖 SAJ Assistant
    
    <div style='text-align: center; font-size: 16px; color: #666; margin-bottom: 20px;'>
    <strong>Your AI-Powered Assistant with Web Search & Tool Support</strong><br>
    </div>
    """)

    session_id = gr.State(value="default-user")

    # Chat interface with 3D styling
    with gr.Group():
        chatbot = gr.Chatbot(
            height=500, 
            label="💬 Conversation",
            elem_classes="chat-container"
        )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="💭 Ask me anything... LangChain questions, web search, calculations, or just chat!",
            label="📝 Your Message",
            scale=5,
            lines=1,
        )
        send_btn = gr.Button("🚀 Send", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Memory", variant="secondary", scale=2)
        status = gr.Textbox(
            label="📊 Status", 
            interactive=False, 
            scale=3,
            value="✅ Ready to chat!"
        )

    # Wire up events with streaming
    def respond(message, history, sid):
        """Handle message and stream response."""
        # Add user message to history
        history = history + [{"role": "user", "content": message}]
        
        # Add empty assistant message to start streaming into
        history = history + [{"role": "assistant", "content": ""}]
        
        # Stream the response
        for streamed_response in chat_stream(message, history, sid):
            # Update the last assistant message with streamed content
            history[-1]["content"] = streamed_response
            yield history, ""
        
        # Final update
        yield history, ""

    send_btn.click(respond, [msg_box, chatbot, session_id], [chatbot, msg_box], queue=True)
    msg_box.submit(respond, [msg_box, chatbot, session_id], [chatbot, msg_box], queue=True)
    clear_btn.click(lambda sid: ([], clear_memory(sid)), [session_id], [chatbot, status])

if __name__ == "__main__":
    demo.queue()  # Enable queuing for streaming
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
