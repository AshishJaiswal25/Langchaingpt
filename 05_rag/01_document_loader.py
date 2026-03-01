"""
Module 5, Exercise 1: Document Loading & Chunking
===================================================
CONCEPT: RAG (Retrieval-Augmented Generation) works in two phases:
  Phase 1 (Indexing):   Load docs → Split into chunks → Store as vectors
  Phase 2 (Retrieval):  User asks → Find relevant chunks → Generate answer

This exercise covers Phase 1 — preparing your documents.

WHY CHUNK? LLMs have context limits. Splitting large docs into
smaller pieces lets you retrieve only the relevant parts.

GOAL: Load documents, split them into chunks, understand the tradeoffs.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------------------------------------------
# STEP 1: Create a sample knowledge base
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Create Sample Documents")
print("=" * 50)

knowledge_base = """
LangChain is a framework for developing applications powered by language models.

Key Features:
1. Chains: Connect multiple LLM calls and tools in sequence
2. Memory: Maintain conversation context across interactions
3. Agents: Enable LLMs to use tools and make decisions
4. RAG: Retrieve and use external knowledge for better answers

LCEL (LangChain Expression Language):
LCEL provides a declarative way to compose chains using the pipe operator (|).
It supports streaming, batch processing, and async execution natively.

Memory Types:
- Buffer Memory: Stores all messages in full
- Summary Memory: Compresses old messages into a summary
- Window Memory: Keeps only the last K messages

RAG Best Practices:
- Use chunk sizes of 500-1000 characters for most documents
- Add 10-20% overlap to avoid cutting context at chunk boundaries
- Choose embedding models that match your language and domain
- Index metadata (source, date, author) alongside content
"""

# Save to disk (like a real document would be)
doc_path = "/tmp/langchain_knowledge.txt"
with open(doc_path, "w") as f:
    f.write(knowledge_base)
print(f"Document saved: {doc_path} ({len(knowledge_base)} chars)")

# -------------------------------------------------------
# STEP 2: Load the document
# -------------------------------------------------------
print("\nSTEP 2: Loading with TextLoader")
print("-" * 30)

loader    = TextLoader(doc_path)
documents = loader.load()

print(f"Documents loaded: {len(documents)}")
print(f"Document type:    {type(documents[0]).__name__}")
print(f"Content preview:  {documents[0].page_content[:100]}...")
print(f"Metadata:         {documents[0].metadata}")

# -------------------------------------------------------
# STEP 3: Split into chunks
# -------------------------------------------------------
print("\nSTEP 3: Splitting into Chunks")
print("-" * 30)

# RecursiveCharacterTextSplitter tries to split on: \n\n, \n, space, ""
# (in that order), preferring natural boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,     # Max characters per chunk
    chunk_overlap=40,   # Characters shared with next chunk
    length_function=len,
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks[:4], 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"  '{chunk.page_content[:80].strip()}...'")

# -------------------------------------------------------
# STEP 4: Chunk size tradeoffs
# -------------------------------------------------------
print("\nSTEP 4: How Chunk Size Affects Results")
print("-" * 30)

for size, overlap in [(100, 20), (300, 60), (600, 100)]:
    s      = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    result = s.split_documents(documents)
    print(f"chunk_size={size:4d}, overlap={overlap:3d} → {len(result):2d} chunks")

# -------------------------------------------------------
# STEP 5: Create documents from strings directly
# -------------------------------------------------------
print("\nSTEP 5: Documents from Strings")
print("-" * 30)

raw_texts = [
    "LangChain makes it easy to build LLM applications.",
    "RAG combines retrieval with generation for accurate answers.",
    "Vector databases store embeddings for fast similarity search.",
]

docs_from_strings = [
    Document(page_content=text, metadata={"source": f"manual_{i}"})
    for i, text in enumerate(raw_texts)
]

for doc in docs_from_strings:
    print(f"[{doc.metadata['source']}] {doc.page_content}")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create your own knowledge base document about a topic you know well
# (e.g. a hobby, a technology, a subject you study)
# Then load and split it with chunk_size=150, chunk_overlap=30
# Print how many chunks it creates and show the first 2 chunks

my_text = """
TODO: Replace this with a multi-paragraph text about any topic.
Write at least 3 paragraphs so the splitter has something to work with.
"""

my_doc     = Document(page_content=my_text)
my_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
my_chunks  = my_splitter.split_documents([my_doc])

print(f"Your text created {len(my_chunks)} chunks")
for i, c in enumerate(my_chunks[:2], 1):
    print(f"Chunk {i}: {c.page_content}")

print("\n✅ Exercise 1 Complete! Move on to 02_retrieval_chain.py")
