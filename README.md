# 🦜 LangChain Hands-On Lab

A complete hands-on project for learning LangChain step by step in VS Code.

## 📁 Folder Structure

```
langchain-lab/
├── 01_prompt_templates/    ← Start here!
│   ├── 01_basic_template.py
│   ├── 02_chat_template.py
│   ├── 03_few_shot_template.py
│   └── 04_advanced_templates.py
│
├── 02_llm_models/
│   ├── 01_first_model.py
│   ├── 02_messages_demo.py
│   ├── 03_model_config.py
│   └── 04_multiple_models.py
│
├── 03_lcel/
│   ├── 01_sequential_chain.py
│   ├── 02_parallel_chains.py
│   ├── 03_dynamic_routing.py
│   └── 04_advanced_lcel.py
│
├── 04_memory/
│   ├── 01_memory_fundamentals.py
│   └── 02_advanced_memory.py
│
├── 05_rag/
│   ├── 01_document_loader.py
│   └── 02_retrieval_chain.py
│
└── 06_chatbot/
    └── app.py               ← Final Gradio chatbot!
```

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run exercises in order
```bash
# Start with Module 1
python 01_prompt_templates/01_basic_template.py

# Then Module 2
python 02_llm_models/01_first_model.py

# ... and so on
```

### 4. Launch the final chatbot
```bash
python 06_chatbot/app.py
# Open http://localhost:7860 in your browser
```

## 📚 Learning Path

| Module | Concept | Key Takeaway |
|--------|---------|--------------|
| 01 | Prompt Templates | Structure your prompts with variables |
| 02 | LLM Models | Connect to and configure AI models |
| 03 | LCEL | Chain components with the `|` operator |
| 04 | Memory | Give your chatbot conversation memory |
| 05 | RAG | Ground AI answers in your documents |
| 06 | Chatbot | Put it all together! |

## 💡 Tips for VS Code
- Install the **Python extension** for syntax highlighting
- Use **Run and Debug** (F5) to step through code
- Open the **Terminal** (Ctrl+`) to run scripts
- Each file has detailed comments explaining what's happening
# Langchaingpt
# LangchainLab
