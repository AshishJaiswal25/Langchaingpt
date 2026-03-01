"""
Module 1, Exercise 3: Few-Shot Templates
=========================================
CONCEPT: Teach the AI by showing examples. Instead of explaining the task,
you demonstrate it 2-5 times, then let the AI continue the pattern.

Pattern:  "happy → sad", "tall → short", "fast → slow" → AI predicts: "hot → cold"

GOAL: Use FewShotPromptTemplate to guide AI behavior with examples.
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# -------------------------------------------------------
# STEP 1: Define your examples
# -------------------------------------------------------
print("=" * 50)
print("STEP 1: Opposites — Teaching by Example")
print("=" * 50)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall",  "output": "short"},
    {"input": "fast",  "output": "slow"},
    {"input": "hot",   "output": "cold"},
]

# Template for how ONE example should look
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# The full few-shot template
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Find the opposite of each word:",
    suffix="Input: {word}\nOutput:",   # The actual question (no answer yet!)
    input_variables=["word"]
)

# See the full prompt that gets sent to the AI
test_word = "big"
full_prompt = few_shot_template.format(word=test_word)
print(f"Full prompt for word '{test_word}':\n")
print(full_prompt)

# -------------------------------------------------------
# STEP 2: Test with multiple words
# -------------------------------------------------------
print("\nSTEP 2: Testing Pattern Learning")
print("-" * 30)

test_words = ["light", "expensive", "difficult", "ancient"]
for word in test_words:
    prompt = few_shot_template.format(word=word)
    # The last line always shows what the AI needs to complete
    last_line = prompt.strip().split("\n")[-1]
    print(f"AI must complete: '{last_line}' (expected: some opposite)")

# -------------------------------------------------------
# STEP 3: Few-shot for a different task — sentiment
# -------------------------------------------------------
print("\nSTEP 3: Sentiment Classification")
print("-" * 30)

sentiment_examples = [
    {"review": "This product is amazing!",      "sentiment": "POSITIVE"},
    {"review": "Worst purchase I've ever made.", "sentiment": "NEGATIVE"},
    {"review": "It's okay, nothing special.",    "sentiment": "NEUTRAL"},
]

sentiment_example_template = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Review: {review}\nSentiment: {sentiment}"
)

sentiment_template = FewShotPromptTemplate(
    examples=sentiment_examples,
    example_prompt=sentiment_example_template,
    prefix="Classify the sentiment of each review:",
    suffix="Review: {review}\nSentiment:",
    input_variables=["review"]
)

new_review = "I love how fast this works but the price is too high."
print(sentiment_template.format(review=new_review))

# -------------------------------------------------------
# 🧪 YOUR TURN: Create a few-shot template for translation
# -------------------------------------------------------
print("\n🧪 YOUR TURN")
print("-" * 30)

# TODO: Create examples for English → French translation
translation_examples = [
    # TODO: Add 3-4 examples like {"english": "hello", "french": "bonjour"}
    {"english": "hello",    "french": "bonjour"},
    {"english": "thank you","french": "merci"},
]

# TODO: Create the example template
translation_example_template = PromptTemplate(
    input_variables=["english", "french"],
    template="English: {english}\nFrench: {french}"
)

# TODO: Build the FewShotPromptTemplate
translation_template = FewShotPromptTemplate(
    examples=translation_examples,
    example_prompt=translation_example_template,
    prefix="Translate English to French:",
    suffix="English: {word}\nFrench:",   # TODO: Adjust suffix
    input_variables=["word"]
)

print(translation_template.format(word="goodbye"))

print("\n✅ Exercise 3 Complete! Move on to 04_advanced_templates.py")
