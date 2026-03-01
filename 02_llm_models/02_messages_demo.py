"""
Module 2, Exercise 2: The Messages System
==========================================
CONCEPT: LLM APIs don't just take plain strings — they take structured messages.
Three types:
  - SystemMessage:  Sets the AI's persona/behavior
  - HumanMessage:   The user's input
  - AIMessage:      The AI's previous replies (for conversation history)

GOAL: Build structured conversations with message history.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,
                   api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# STEP 1: Messages vs plain strings
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Plain String vs Structured Messages")
print("=" * 50)

# Plain string — works but gives no control over behavior
plain_response = model.invoke("Explain lists in Python")
print(f"Plain string response:\n{plain_response.content[:150]}...\n")

# Structured messages — you control the persona
messages = [
    SystemMessage(content="You are a Python tutor who explains everything with "
                           "emoji and keeps answers under 3 sentences."),
    HumanMessage(content="Explain lists in Python"),
]

structured_response = model.invoke(messages)
print(f"Structured message response:\n{structured_response.content}\n")

# -------------------------------------------------------
# STEP 2: Building a multi-turn conversation
# -------------------------------------------------------
print("STEP 2: Multi-Turn Conversation with Memory")
print("-" * 30)

# The chat history is just a Python list — you manage it manually
chat_history = [
    SystemMessage(content="You are a friendly assistant. Remember everything the user tells you.")
]

# Turn 1
user_input = "My name is Alice and I'm learning LangChain."
chat_history.append(HumanMessage(content=user_input))

response = model.invoke(chat_history)
chat_history.append(response)  # ← Add AI reply to history!

print(f"User: {user_input}")
print(f"AI:   {response.content}\n")

# Turn 2 — AI should remember the name
user_input2 = "What's my name and what am I learning?"
chat_history.append(HumanMessage(content=user_input2))

response2 = model.invoke(chat_history)
chat_history.append(response2)

print(f"User: {user_input2}")
print(f"AI:   {response2.content}\n")

# Turn 3
user_input3 = "Give me one tip for learning it faster."
chat_history.append(HumanMessage(content=user_input3))

response3 = model.invoke(chat_history)
print(f"User: {user_input3}")
print(f"AI:   {response3.content}\n")

# -------------------------------------------------------
# STEP 3: Inspecting the history
# -------------------------------------------------------
print("STEP 3: Full Conversation History")
print("-" * 30)

for i, msg in enumerate(chat_history):
    label = {
        "system": "🔵 SYSTEM",
        "human":  "🟢 USER  ",
        "ai":     "🟣 AI    ",
    }.get(msg.type, "⚪ OTHER ")
    print(f"[{i}] {label}: {msg.content[:70]}...")

# -------------------------------------------------------
# 🧪 YOUR TURN: Build your own conversation
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a conversation with a custom persona
# The AI should be a fitness coach who remembers your goals

my_history = [
    SystemMessage(content="TODO: Write a system prompt for a fitness coach persona"),
]

# TODO: Add at least 2 HumanMessages and capture the AI responses
# Hint: Follow the pattern from STEP 2

print("\n✅ Exercise 2 Complete! Move on to 03_model_config.py")
