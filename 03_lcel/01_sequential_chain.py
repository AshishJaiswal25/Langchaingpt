"""
Module 3, Exercise 1: Sequential Chains with LCEL
===================================================
CONCEPT: LCEL (LangChain Expression Language) uses the pipe operator |
to chain components together. Data flows left to right:

  prompt | model | parser

Each component receives the output of the previous one.

GOAL: Build your first LCEL chain and understand how data flows.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

model  = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()  # Converts AIMessage → plain string

# -------------------------------------------------------
# STEP 1: Your first ch
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Your First LCEL Chain")
print("=" * 50)

prompt = ChatPromptTemplate.from_template(
    "Answer this question concisely: {question}"
)

# THE CHAIN: prompt | model | parser
# 1. prompt formats the input dict into messages
# 2. model sends messages to the LLM and gets an AIMessage back
# 3. parser extracts the .content string from AIMessage
chain = prompt | model | parser

result = chain.invoke({"question": "What is LCEL in LangChain?"})
print(f"Result type: {type(result).__name__}")  # str, not AIMessage!
print(f"Result: {result}")

# -------------------------------------------------------
# STEP 2: What happens at each stage?
# -------------------------------------------------------
print("\nSTEP 2: Inspecting Each Stage")
print("-" * 30)

# Stage 1: Just the prompt
formatted = prompt.invoke({"question": "What is 2+2?"})
print(f"After prompt: {type(formatted).__name__}")
print(f"  Messages: {formatted.messages}")

# Stage 2: Prompt → model (get AIMessage)
ai_msg = (prompt | model).invoke({"question": "What is 2+2?"})
print(f"\nAfter model: {type(ai_msg).__name__}")
print(f"  Content: {ai_msg.content}")

# Stage 3: Add parser (get string)
string_result = (prompt | model | parser).invoke({"question": "What is 2+2?"})
print(f"\nAfter parser: {type(string_result).__name__}")
print(f"  Result: {string_result}")

# -------------------------------------------------------
# STEP 3: Multi-step chain (chain outputs feed into next chain)
# -------------------------------------------------------
print("\nSTEP 3: Chaining Two LLM Calls")
print("-" * 30)

# Chain A: Generate a topic
topic_prompt = ChatPromptTemplate.from_template(
    "Suggest one creative research topic about: {subject}"
)

# Chain B: Write about that topic
essay_prompt = ChatPromptTemplate.from_template(
    "Write a 2-sentence explanation of: {topic}"
)

# Connect them: output of chain A becomes input to chain B
full_chain = (
    {"topic": topic_prompt | model | parser}
    | essay_prompt
    | model
    | parser
)

essay = full_chain.invoke({"subject": "artificial intelligence"})
print(f"Generated essay:\n{essay}")

# -------------------------------------------------------
# STEP 4: RunnablePassthrough — pass input through unchanged
# -------------------------------------------------------
print("\nSTEP 4: RunnablePassthrough")
print("-" * 30)

# Sometimes you want BOTH the original input AND the AI response
chain_with_input = RunnableParallel(
    original_question=RunnablePassthrough(),  # Pass input through
    answer=(prompt | model | parser),          # Also run through chain
)

result = chain_with_input.invoke({"question": "What is Python?"})
print(f"Original: {result['original_question']}")
print(f"Answer:   {result['answer']}")

# -------------------------------------------------------
# 🧪 YOUR TURN: Build a two-step chain
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Build a chain that:
# 1. Takes a {word} as input
# 2. First generates a definition of that word
# 3. Then generates an example sentence using that word

step1_prompt = ChatPromptTemplate.from_template(
    "Define this word in one sentence: {word}"
)

step2_prompt = ChatPromptTemplate.from_template(
    "Write an example sentence using this concept: {definition}"
)

# TODO: Connect them into a chain
# my_chain = {"definition": step1_prompt | model | parser} | step2_prompt | model | parser
# result = my_chain.invoke({"word": "recursion"})
# print(result)

print("\n✅ Exercise 1 Complete! Move on to 02_parallel_chains.py")
