"""
Module 4, Exercise 1: Memory Fundamentals
==========================================
CONCEPT: LLMs are stateless — they forget everything between calls.
Memory systems solve this by storing and replaying conversation history.

RunnableWithMessageHistory wraps your chain and automatically:
  - Loads history before each call
  - Saves new messages after each call

GOAL: Give your chain persistent memory across multiple turns.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,
                   api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# STEP 1: Without memory — the AI forgets
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: No Memory (Stateless)")
print("=" * 50)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

simple_chain = (
    ChatPromptTemplate.from_template("Answer: {question}")
    | model
    | StrOutputParser()
)

r1 = simple_chain.invoke({"question": "My name is Alice."})
print(f"Turn 1: {r1}")

r2 = simple_chain.invoke({"question": "What's my name?"})
print(f"Turn 2: {r2}")  # Will say it doesn't know — no memory!

# -------------------------------------------------------
# STEP 2: Add memory with RunnableWithMessageHistory
# -------------------------------------------------------
print("\nSTEP 2: With Memory")
print("-" * 30)

# The history store: session_id → InMemoryChatMessageHistory
# In production, replace with a database-backed store
history_store: dict = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) the history for this session."""
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# The prompt now includes a {history} placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember what the user tells you."),
    MessagesPlaceholder(variable_name="history"),  # ← History injected here
    ("human", "{input}"),
])

chain = prompt | model

# Wrap the chain with history management
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,                     # Function that returns history object
    input_messages_key="input",      # Which key holds the user's new message
    history_messages_key="history",  # Which key holds past messages
)

# config identifies the session (user/conversation)
config = {"configurable": {"session_id": "alice-session"}}

# Now the chain remembers!
t1 = chain_with_memory.invoke({"input": "My name is Alice and I work as a data scientist."}, config)
print(f"Turn 1: {t1.content}")

t2 = chain_with_memory.invoke({"input": "What's my name and what do I do?"}, config)
print(f"Turn 2: {t2.content}")

t3 = chain_with_memory.invoke({"input": "Give me a tip related to my job."}, config)
print(f"Turn 3: {t3.content}")

# -------------------------------------------------------
# STEP 3: Inspect what's in memory
# -------------------------------------------------------
print("\nSTEP 3: Inspecting the Memory Store")
print("-" * 30)

history = history_store.get("alice-session")
print(f"Messages in memory: {len(history.messages)}")
for msg in history.messages:
    label = "USER" if msg.type == "human" else "AI  "
    print(f"  [{label}] {msg.content[:60]}...")

# -------------------------------------------------------
# STEP 4: Multiple users — separate sessions
# -------------------------------------------------------
print("\nSTEP 4: Multiple Sessions (Multiple Users)")
print("-" * 30)

config_bob = {"configurable": {"session_id": "bob-session"}}

chain_with_memory.invoke({"input": "I'm Bob and I love hiking."}, config_bob)
r_bob  = chain_with_memory.invoke({"input": "What do I love?"}, config_bob)
r_alice = chain_with_memory.invoke({"input": "What's my hobby?"}, config)  # Alice's session

print(f"Bob's  hobby: {r_bob.content}")
print(f"Alice's hobby: {r_alice.content}")   # Should mention data science, not hiking!

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a new session "student-session" where:
# 1. Tell the AI your name and your favourite subject
# 2. Ask it to quiz you on that subject
# 3. Verify it remembered your name when responding

# config_student = {"configurable": {"session_id": "student-session"}}
# chain_with_memory.invoke({"input": "..."}, config_student)

print("\n✅ Exercise 1 Complete! Move on to 02_advanced_memory.py")
