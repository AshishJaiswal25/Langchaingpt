"""
Module 1, Exercise 1: Basic Prompt Templates
=============================================
CONCEPT: Templates are like fill-in-the-blank sentences.
You define placeholders like {variable} and fill them in later.

GOAL: Understand how PromptTemplate works and why it's useful.
"""

from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------
# STEP 1: Create a basic template
# -------------------------------------------------------
# {product} and {feature} are placeholders — they'll be filled in later
template = PromptTemplate(
    input_variables=["product", "feature"],
    template="Generate a marketing slogan for {product} highlighting {feature}."
)

print("=" * 50)
print("STEP 1: Basic Template")
print("=" * 50)

# Fill in the template with actual values
prompt = template.format(product="LangChain", feature="AI orchestration")
print("Generated prompt:", prompt)

# -------------------------------------------------------
# STEP 2: Reuse the same template with different values
# -------------------------------------------------------
print("\nSTEP 2: Reusing the Template")
print("-" * 30)

examples = [
    {"product": "Smartphone", "feature": "camera quality"},
    {"product": "Electric Car", "feature": "eco-friendly design"},
    {"product": "AI Assistant", "feature": "natural conversation"},
]

for example in examples:
    result = template.format(**example)
    print(f"• {result}")

# -------------------------------------------------------
# STEP 3: Inspect the template
# -------------------------------------------------------
print("\nSTEP 3: Template Inspection")
print("-" * 30)
print(f"Input variables: {template.input_variables}")
print(f"Template string: {template.template}")

# -------------------------------------------------------
# 🧪 YOUR TURN: Try it yourself!
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create your own template for a job posting
# It should have placeholders for: job_title, skill, company
my_template = PromptTemplate(
    input_variables=["job_title", "skill", "company"],
    template="TODO: Write your template here with {job_title}, {skill}, and {company}"
)

# TODO: Fill in your template with values
my_prompt = my_template.format(
    job_title="Software Engineer",
    skill="Python",
    company="OpenAI"
)
print("Your prompt:", my_prompt)

print("\n✅ Exercise 1 Complete! Move on to 02_chat_template.py")
