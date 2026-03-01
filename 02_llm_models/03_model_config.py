"""
Module 2, Exercise 3: Model Configuration & Streaming
======================================================
CONCEPT: Key parameters that control model behavior:
  - temperature:  0.0 = focused/deterministic, 1.0 = creative/random
  - max_tokens:   Limits response length (controls cost too)
  - streaming:    Stream tokens as they're generated (feels faster to users)

GOAL: Understand how config parameters change output, and implement streaming.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------------
# STEP 1: Temperature — the creativity dial
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Temperature Comparison")
print("=" * 50)

precise_model  = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=API_KEY)
balanced_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=API_KEY)
creative_model = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=API_KEY)

prompt = "Describe a sunset in one sentence."

print(f"Prompt: '{prompt}'\n")
print(f"[temp=0.0 — precise]  {precise_model.invoke(prompt).content}")
print(f"[temp=0.5 — balanced] {balanced_model.invoke(prompt).content}")
print(f"[temp=1.0 — creative] {creative_model.invoke(prompt).content}")

# Run the creative one twice — notice the variation!
print(f"\n[temp=1.0 again]      {creative_model.invoke(prompt).content}")

# -------------------------------------------------------
# STEP 2: max_tokens — controlling response length
# -------------------------------------------------------
print("\nSTEP 2: Controlling Response Length")
print("-" * 30)

short_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, 
                          max_tokens=20, api_key=API_KEY)
long_model  = ChatOpenAI(model="gpt-4o-mini", temperature=0, 
                          max_tokens=200, api_key=API_KEY)

question = "What is machine learning?"
print(f"Short response (max 20 tokens):  {short_model.invoke(question).content}")
print(f"Long response  (max 200 tokens): {long_model.invoke(question).content[:150]}...")

# -------------------------------------------------------
# STEP 3: Streaming — get tokens as they arrive
# -------------------------------------------------------
print("\nSTEP 3: Streaming Response")
print("-" * 30)

streaming_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=API_KEY,
    streaming=True,
)

print("Streaming: ", end="", flush=True)
full_response = ""
for chunk in streaming_model.stream("Write a haiku about coding"):
    print(chunk.content, end="", flush=True)
    full_response += chunk.content
print()  # Newline after streaming ends

print(f"\nFull response collected: {repr(full_response)}")

# -------------------------------------------------------
# STEP 4: Use case guide
# -------------------------------------------------------
print("\nSTEP 4: Which Settings for Which Use Cases?")
print("-" * 30)

use_cases = [
    ("Fact lookup / math",        0.0, 100),
    ("Summarization",             0.2, 300),
    ("Chatbot conversation",      0.7, 500),
    ("Creative writing",          0.9, 800),
    ("Brainstorming ideas",       1.0, 400),
]

print(f"{'Use Case':<30} {'temp':>6} {'max_tokens':>12}")
print("-" * 50)
for case, temp, tokens in use_cases:
    print(f"{case:<30} {temp:>6.1f} {tokens:>12}")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create two models with different temperatures
# Then ask both the same open-ended question and compare
# Question idea: "What are 3 uses for AI in healthcare?"

model_a = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=API_KEY)
model_b = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=API_KEY)

question = "TODO: Replace with your question"
# print(f"Model A: {model_a.invoke(question).content}")
# print(f"Model B: {model_b.invoke(question).content}")

print("\n✅ Exercise 3 Complete! Move on to 04_multiple_models.py")
