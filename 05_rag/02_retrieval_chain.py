"""
Module 5, Exercise 2: Full RAG Chain
======================================
CONCEPT: Now we combine everything:
  Documents → Embeddings → FAISS Vector Store → Retriever
  → LCEL RAG Chain → Contextual Answer

Embeddings convert text into numerical vectors so that
"What is LCEL?" and "LangChain pipe operator" end up close
together in vector space — even though the words differ.

GOAL: Build a complete RAG system that answers questions
      using your document as the source of truth.

SETUP: pip install langchain-huggingface faiss-cpu sentence-transformers
"""

import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                   api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# STEP 1: Load free HuggingFace embeddings
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Loading Embeddings (no API key needed!)")
print("=" * 50)

# sentence-transformers runs locally on CPU — completely free
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Test the embeddings
test_vector = embeddings.embed_query("LangChain")
print(f"Embedding dimensions: {len(test_vector)}")  # 384 dims
print(f"First 5 values: {[round(v, 4) for v in test_vector[:5]]}")

# -------------------------------------------------------
# STEP 2: Build the vector store
# -------------------------------------------------------
print("\nSTEP 2: Building FAISS Vector Store")
print("-" * 30)

sample_docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "intro.txt", "topic": "overview"}
    ),
    Document(
        page_content="LCEL provides a declarative way to compose chains using the pipe (|) operator.",
        metadata={"source": "lcel.txt", "topic": "lcel"}
    ),
    Document(
        page_content="Memory systems in LangChain help maintain conversation context across turns.",
        metadata={"source": "memory.txt", "topic": "memory"}
    ),
    Document(
        page_content="RAG combines retrieval with generation for accurate, grounded responses.",
        metadata={"source": "rag.txt", "topic": "rag"}
    ),
    Document(
        page_content="Best practice: use chunk sizes of 500-1000 characters with 10-20% overlap.",
        metadata={"source": "best_practices.txt", "topic": "rag"}
    ),
    Document(
        page_content="Agents in LangChain can use tools like search engines and calculators to solve tasks.",
        metadata={"source": "agents.txt", "topic": "agents"}
    ),
]

# Split docs for cleaner retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(sample_docs)

# Build FAISS vector store from documents
vectorstore = FAISS.from_documents(split_docs, embeddings)
print(f"Vector store created with {len(split_docs)} chunks")

# -------------------------------------------------------
# STEP 3: Retriever — find relevant chunks
# -------------------------------------------------------
print("\nSTEP 3: Retriever")
print("-" * 30)

# k=3: retrieve the top-3 most similar chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test retrieval independently
query = "What is LCEL?"
retrieved = retriever.invoke(query)
print(f"Query: '{query}'")
print(f"Retrieved {len(retrieved)} chunks:")
for i, doc in enumerate(retrieved, 1):
    print(f"  [{i}] ({doc.metadata.get('topic', '?')}) {doc.page_content[:70]}...")

# -------------------------------------------------------
# STEP 4: Build the LCEL RAG chain
# -------------------------------------------------------
print("\nSTEP 4: RAG Chain")
print("-" * 30)

def format_docs(docs):
    """Join retrieved chunks into one context string."""
    return "\n\n".join(f"[{doc.metadata.get('source','?')}]\n{doc.page_content}"
                       for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use ONLY the following context to answer the question.\n"
     "If the answer isn't in the context, say 'I don't have that information.'\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# The chain:
# 1. Retrieve docs for the question
# 2. Format them into a context string
# 3. Inject context + question into the prompt
# 4. Generate an answer
rag_chain = (
    RunnableParallel({
        "context":  retriever | RunnablePassthrough() ,
        "question": RunnablePassthrough(),
    })
    .assign(formatted_context=lambda x: format_docs(x["context"]))
    .assign(
        answer=(
            {
                "context":  itemgetter("formatted_context"),
                "question": itemgetter("question"),
            }
            | rag_prompt
            | model
            | StrOutputParser()
        )
    )
    .pick(["answer", "context"])
)

# -------------------------------------------------------
# STEP 5: Test the RAG system
# -------------------------------------------------------
print("\nSTEP 5: Question Answering")
print("=" * 50)

questions = [
    "What is LCEL and how does it work?",
    "How do memory systems work in LangChain?",
    "What is the recommended chunk size for RAG?",
    "Can LangChain agents use external tools?",
    "What is the capital of France?",   # Not in the docs — should say "I don't know"
]

for q in questions:
    result = rag_chain.invoke(q)
    print(f"\n❓ {q}")
    print(f"💡 {result['answer']}")
    sources = {doc.metadata.get('source', '?') for doc in result['context']}
    print(f"📚 Sources: {', '.join(sources)}")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Add 3 more documents about a topic you choose
# Then ask 2 questions that can only be answered with those docs

my_docs = [
    Document(page_content="TODO: Write your first document here.",
             metadata={"source": "mine_1.txt"}),
    # Add more docs...
]

# Add them to the existing vector store (incremental update)
vectorstore.add_documents(my_docs)

# Now ask a question that only your docs can answer
# result = rag_chain.invoke("TODO: Your question here")
# print(result["answer"])

print("\n✅ Module 5 Complete! Move on to 06_chatbot/app.py")
