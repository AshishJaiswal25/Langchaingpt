"""
Module 2, Exercise 1: Your First LLM Model
===========================================
CONCEPT: ChatOpenAI is LangChain's wrapper around OpenAI (and compatible) APIs.
You initialize it once, then call .invoke() to send messages.

GOAL: Connect to an LLM and get your first response.

SETUP REQUIRED:
  1. Copy .env.example to .env
  2. Add your OPENAI_API_KEY to .env
  3. Run: pip install python-dotenv langchain-openai
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------
# STEP 1: Initialize the model
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Initialize ChatOpenAI")
print("=" * 50)

model = ChatOpenAI(
    model="gpt-4o-mini",        # The model to use
    temperature=0,              # 0 = deterministic, 1 = creative
    api_key=os.getenv("OPENAI_API_KEY"),   # From your .env file
)

print(f"Model initialized: {model.model_name}")

# -------------------------------------------------------
# STEP 2: Send your first message
# -------------------------------------------------------
print("\nSTEP 2: First Message")
print("-" * 30)

response = model.invoke("Hello! What is LangChain in one sentence?")

# The response is an AIMessage object
print(f"Response type: {type(response).__name__}")
print(f"Content: {response.content}")

# -------------------------------------------------------
# STEP 3: Inspect the response object
# -------------------------------------------------------
print("\nSTEP 3: Exploring the Response Object")
print("-" * 30)

response2 = model.invoke("What's 15 * 8?")
print(f"Content:    {response2.content}")
print(f"Model used: {response2.response_metadata.get('model_name', 'N/A')}")

# Token usage (important for cost management!)
usage = response2.usage_metadata
if usage:
    print(f"Tokens used — Input: {usage.get('input_tokens')}, "
          f"Output: {usage.get('output_tokens')}")

# -------------------------------------------------------
# STEP 4: Quick Q&A loop
# -------------------------------------------------------
print("\nSTEP 4: Multiple Questions")
print("-" * 30)

questions = [
    "Name three Python data structures.",
    "What does API stand for?",
    "Explain prompt engineering in one sentence.",
]

for q in questions:
    answer = model.invoke(q)
    print(f"Q: {q}")
    print(f"A: {answer.content}\n")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("🧪 YOUR TURN")
print("-" * 30)

# TODO: Ask the model a question about LangChain
# Replace the question below with something you're curious about
my_question = "TODO: Replace this with your own question"
my_response = model.invoke(my_question)
print(f"Q: {my_question}")
print(f"A: {my_response.content}")

print("\n✅ Exercise 1 Complete! Move on to 02_messages_demo.py")
