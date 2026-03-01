"""
Module 1, Exercise 4: Advanced Templates
==========================================
CONCEPT: Production templates need:
  - Partial variables (pre-filled constants)
  - Structured output (Pydantic models)
  - Conditional logic (different templates per user level)

GOAL: Build robust, reusable templates for real applications.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# -------------------------------------------------------
# STEP 1: Partial Templates (pre-fill some variables)
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Partial Variables")
print("=" * 50)

# This template has TWO variables: {role} and {task}
# We pre-fill {role} so the user only needs to provide {task}
full_template = PromptTemplate(
    input_variables=["task"],
    template="As an expert {role}, please help with: {task}",
    partial_variables={"role": "AI consultant"}  # Pre-filled!
)

# User only needs to supply {task}
prompt = full_template.format(task="optimize database queries")
print(f"Prompt: {prompt}")

# You can also partially fill AFTER creation
template2 = PromptTemplate(
    input_variables=["role", "task"],
    template="As a {role}, help me with: {task}"
)
data_template = template2.partial(role="data scientist")  # Lock in the role
print(f"Data prompt: {data_template.format(task='build a model')}")

# -------------------------------------------------------
# STEP 2: Structured Output with Pydantic
# -------------------------------------------------------
print("\nSTEP 2: Structured Output")
print("-" * 30)

# Define the SHAPE of the output you expect
class ProductReview(BaseModel):
    rating:         int       = Field(description="Rating from 1 to 5")
    pros:           list[str] = Field(description="List of advantages")
    cons:           list[str] = Field(description="List of disadvantages")
    recommendation: str       = Field(description="Final recommendation in one sentence")

# Create a parser that enforces this shape
parser = PydanticOutputParser(pydantic_object=ProductReview)

# The parser generates instructions for the AI automatically!
format_instructions = parser.get_format_instructions()
print("Auto-generated format instructions (first 200 chars):")
print(format_instructions[:200] + "...\n")

# Embed those instructions in the template
structured_template = PromptTemplate(
    template="Review this product: {product}\n\n{format_instructions}",
    input_variables=["product"],
    partial_variables={"format_instructions": format_instructions}
)

prompt = structured_template.format(product="iPhone 15 Pro")
print(f"Structured prompt (first 300 chars):\n{prompt[:300]}...")

# -------------------------------------------------------
# STEP 3: Conditional Templates
# -------------------------------------------------------
print("\nSTEP 3: Conditional Templates")
print("-" * 30)

def create_template_for_level(user_level: str) -> PromptTemplate:
    """Return a different template based on user's expertise level."""
    templates = {
        "beginner":     "Explain {concept} using a simple analogy. Avoid jargon.",
        "intermediate": "Explain {concept} with some technical detail and an example.",
        "advanced":     "Give a comprehensive technical explanation of {concept}, "
                        "including edge cases and best practices.",
    }
    selected = templates.get(user_level, templates["beginner"])
    return PromptTemplate(input_variables=["concept"], template=selected)

for level in ["beginner", "intermediate", "advanced"]:
    t = create_template_for_level(level)
    prompt = t.format(concept="recursion")
    print(f"[{level.upper()}] {prompt}\n")

# -------------------------------------------------------
# 🧪 YOUR TURN: Combine partial variables + conditional
# -------------------------------------------------------
print("🧪 YOUR TURN")
print("-" * 30)

# TODO: Create a template for a customer support bot
# - Pre-fill the company name with partial_variables
# - Accept {issue} as user input

support_template = PromptTemplate(
    input_variables=["issue"],
    template="You are a support agent for {company}. Help the customer with: {issue}",
    partial_variables={"company": "TODO: Your Company Name"}
)

print(support_template.format(issue="my order hasn't arrived"))

print("\n✅ Module 1 Complete! Move to 02_llm_models/")
