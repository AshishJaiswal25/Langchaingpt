"""
Module 3, Exercise 3: Dynamic Routing with RunnableLambda
==========================================================
CONCEPT: RunnableLambda lets you wrap any Python function as a chain step.
Use it to:
  - Transform data between steps
  - Route to different chains based on conditions
  - Add custom business logic inside a chain

GOAL: Build intelligent chains that choose their own path.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model  = ChatOpenAI(model="gpt-4o-mini", temperature=0.5,
                    api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# -------------------------------------------------------
# STEP 1: Basic data transformation with RunnableLambda
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Data Transformation")
print("=" * 50)

def clean_text(text: str) -> str:
    """Remove extra whitespace and make lowercase."""
    return " ".join(text.lower().split())

def add_word_count(text: str) -> Dict:
    """Convert text to dict with metadata."""
    return {
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
    }

# Chain two lambda functions together
transform_chain = (
    RunnableLambda(clean_text)
    | RunnableLambda(add_word_count)
)

result = transform_chain.invoke("  Hello   World   from  LangChain!  ")
print(f"Transformed: {result}")

# -------------------------------------------------------
# STEP 2: Conditional routing based on input length
# -------------------------------------------------------
print("\nSTEP 2: Route by Input Length")
print("-" * 30)

short_prompt = ChatPromptTemplate.from_template(
    "Give a one-line answer: {text}"
)
long_prompt = ChatPromptTemplate.from_template(
    "Give a detailed analysis of: {text}"
)

def route_by_length(input_dict: Dict) -> str:
    """Use a short or detailed prompt depending on input length."""
    text = input_dict.get("text", "")
    if len(text) < 30:
        chain = short_prompt | model | parser
    else:
        chain = long_prompt  | model | parser
    return chain.invoke(input_dict)

router = RunnableLambda(route_by_length)

print("Short input:")
print(router.invoke({"text": "Hi!"}))

print("\nLong input:")
print(router.invoke({"text": "Explain the long-term societal impacts of generative AI on the job market"}))

# -------------------------------------------------------
# STEP 3: Multi-condition routing by query type
# -------------------------------------------------------
print("\nSTEP 3: Smart Query Router")
print("-" * 30)

def classify_query(text: str) -> str:
    """Classify what kind of query this is."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["code", "python", "function", "bug", "error"]):
        return "technical"
    elif any(k in text_lower for k in ["explain", "what is", "how does", "why"]):
        return "educational"
    elif any(k in text_lower for k in ["joke", "funny", "humor", "laugh"]):
        return "entertainment"
    return "general"

def smart_route(input_dict: Dict) -> str:
    query      = input_dict.get("query", "")
    query_type = classify_query(query)

    prompts = {
        "technical":     "Provide a technical answer with a short code example: {query}",
        "educational":   "Explain clearly and simply: {query}",
        "entertainment": "Respond humorously: {query}",
        "general":       "Answer helpfully: {query}",
    }

    prompt  = ChatPromptTemplate.from_template(prompts[query_type])
    chain   = prompt | model | parser
    answer  = chain.invoke(input_dict)
    return f"[{query_type.upper()}] {answer}"

smart_router = RunnableLambda(smart_route)

test_queries = [
    "Write a Python function to sort a list",
    "What is a neural network?",
    "Tell me something funny about debugging",
    "What's the capital of Japan?",
]

for q in test_queries:
    print(f"\nQ: {q}")
    print(f"A: {smart_router.invoke({'query': q})[:150]}...")

# -------------------------------------------------------
# 🧪 YOUR TURN
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a router that handles 3 topics:
# - "recipe": Give me a simple recipe for {food}
# - "history": Tell me a historical fact about {topic}
# - "default": Just answer: {query}

# Hint: Classify based on keywords in the input dict
# Then select and invoke the right chain

def my_router(input_dict: Dict) -> str:
    query = input_dict.get("query", "")
    # TODO: Add your classification and routing logic here
    return f"TODO: Route this → '{query}'"

my_chain = RunnableLambda(my_router)
print(my_chain.invoke({"query": "Tell me a recipe for pasta"}))

print("\n✅ Exercise 3 Complete! Move on to 04_advanced_lcel.py")
