"""
Module 1, Exercise 2: Chat Prompt Templates
============================================
CONCEPT: Chat models (like GPT-4) work with structured messages, not just strings.
There are 3 message types:
  - system:    Instructions for the AI (sets behavior)
  - human:     What the user says
  - assistant: What the AI says (for multi-turn context)

GOAL: Build structured conversations using ChatPromptTemplate.
"""

from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------
# STEP 1: Create a simple chat template
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Simple Chat Template")
print("=" * 50)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert with {years} years of experience."),
    ("human",  "Explain {concept} to me in simple terms."),
])

# Generate the messages (not yet sent to AI — just formatted)
messages = chat_template.format_messages(
    role="Python programming",
    years="10",
    concept="decorators"
)

# Print each message with a visual indicator
icons = {"system": "🔵 SYSTEM", "human": "🟢 HUMAN", "ai": "🟣 AI"}
for msg in messages:
    icon = icons.get(msg.type, "⚪ UNKNOWN")
    print(f"{icon}: {msg.content}")

# -------------------------------------------------------
# STEP 2: Include an assistant message (few-shot style)
# -------------------------------------------------------
print("\nSTEP 2: Including Assistant Messages")
print("-" * 30)

# Adding an assistant message shows the AI how to respond
few_shot_template = ChatPromptTemplate.from_messages([
    ("system",    "You are a helpful coding tutor."),
    ("human",     "What is a variable?"),
    ("assistant", "A variable is like a labeled box that stores a value, like x = 5."),  # Example
    ("human",     "Now explain {concept}."),  # The real question
])

messages = few_shot_template.format_messages(concept="functions")
print(f"Total messages created: {len(messages)}")
for msg in messages:
    print(f"  [{msg.type}]: {msg.content[:60]}...")

# -------------------------------------------------------
# STEP 3: Different scenarios with the same template
# -------------------------------------------------------
print("\nSTEP 3: Reusing Template Across Scenarios")
print("-" * 30)

scenarios = [
    {"role": "Data Science",    "years": "5",  "concept": "machine learning"},
    {"role": "Web Development", "years": "8",  "concept": "REST APIs"},
    {"role": "Cybersecurity",   "years": "12", "concept": "SQL injection"},
]

for scenario in scenarios:
    msgs = chat_template.format_messages(**scenario)
    system_msg = msgs[0].content
    print(f"🔵 {system_msg}")

# -------------------------------------------------------
# 🧪 YOUR TURN: Build your own chat template
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a chat template for a language tutor
# System: You are a {language} tutor
# Human: How do I say "{phrase}" in {language}?
my_template = ChatPromptTemplate.from_messages([
    # TODO: Add your messages here
    ("system", "You are a {language} language tutor."),
    ("human",  "TODO: Complete this message about {phrase}"),
])

my_messages = my_template.format_messages(language="Spanish", phrase="good morning")
for msg in my_messages:
    print(f"[{msg.type}]: {msg.content}")

print("\n✅ Exercise 2 Complete! Move on to 03_few_shot_template.py")
