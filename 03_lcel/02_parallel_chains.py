"""
Module 3, Exercise 2: Parallel Chains
=======================================
CONCEPT: RunnableParallel runs multiple chains SIMULTANEOUSLY.
Instead of waiting for chain A to finish before starting chain B,
both run at the same time — cutting total latency.

  Input ──┬── Chain A ──┐
          ├── Chain B ──┤── Combined Output
          └── Chain C ──┘

GOAL: Use RunnableParallel for speed and multi-perspective analysis.
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model  = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,
                    api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# -------------------------------------------------------
# STEP 1: Basic parallel execution
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Three Chains in Parallel")
print("=" * 50)

joke_prompt  = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
fact_prompt  = ChatPromptTemplate.from_template("Share one surprising fact about {topic}")
quote_prompt = ChatPromptTemplate.from_template("Write an inspiring quote about {topic}")

joke_chain  = joke_prompt  | model | parser
fact_chain  = fact_prompt  | model | parser
quote_chain = quote_prompt | model | parser

# RunnableParallel runs all three at the same time
parallel_chain = RunnableParallel(
    joke=joke_chain,
    fact=fact_chain,
    quote=quote_chain,
)

start = time.time()
results = parallel_chain.invoke({"topic": "programming"})
elapsed = time.time() - start

print(f"⏱  All 3 chains completed in {elapsed:.2f}s\n")
print(f"😄 Joke:  {results['joke']}")
print(f"📚 Fact:  {results['fact']}")
print(f"✨ Quote: {results['quote']}")

# -------------------------------------------------------
# STEP 2: Text analysis — multiple perspectives at once
# -------------------------------------------------------
print("\nSTEP 2: Parallel Text Analysis")
print("-" * 30)

text = "LangChain makes building LLM applications easy and fun!"

analysis_chain = RunnableParallel(
    sentiment=ChatPromptTemplate.from_template(
        "What is the sentiment (positive/negative/neutral) of: '{text}'? "
        "Answer in 5 words max."
    ) | model | parser,

    summary=ChatPromptTemplate.from_template(
        "Summarize in 8 words: '{text}'"
    ) | model | parser,

    keywords=ChatPromptTemplate.from_template(
        "List 3 keywords from: '{text}'. Format: word1, word2, word3"
    ) | model | parser,
)

analysis = analysis_chain.invoke({"text": text})
print(f"Text:      '{text}'")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Summary:   {analysis['summary']}")
print(f"Keywords:  {analysis['keywords']}")

# -------------------------------------------------------
# STEP 3: Parallel then sequential (gather, then summarize)
# -------------------------------------------------------
print("\nSTEP 3: Parallel → Sequential")
print("-" * 30)

# First: gather pros and cons in parallel
gather = RunnableParallel(
    pros=ChatPromptTemplate.from_template("List 2 pros of {topic}") | model | parser,
    cons=ChatPromptTemplate.from_template("List 2 cons of {topic}") | model | parser,
)

# Then: use both results in a summary (sequential)
summary_prompt = ChatPromptTemplate.from_template(
    "Pros:\n{pros}\n\nCons:\n{cons}\n\nWrite a balanced 1-sentence conclusion."
)

combined_chain = gather | summary_prompt | model | parser
conclusion = combined_chain.invoke({"topic": "remote work"})
print(f"Balanced conclusion on remote work:\n{conclusion}")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a parallel chain that analyses a book/movie from 3 angles:
# 1. themes: What are the main themes?
# 2. audience: Who is the target audience?
# 3. one_liner: Write a one-line pitch for it.

# book_analysis = RunnableParallel(
#     themes=...,
#     audience=...,
#     one_liner=...,
# )
# result = book_analysis.invoke({"title": "The Hobbit"})
# print(result)

print("\n✅ Exercise 2 Complete! Move on to 03_dynamic_routing.py")
