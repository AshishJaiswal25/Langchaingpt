# 🌐 Google Search Integration Setup

Your chatbot now supports **Google Search** to find real-time information when your knowledge base doesn't have the answer!

## Setup Instructions

### 1. Get Google API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the **Custom Search API**
4. Create an API key in **Credentials**
5. Copy your `GOOGLE_API_KEY`

### 2. Get Custom Search Engine ID

1. Go to [Google Custom Search](https://cse.google.com/cse/)
2. Create a new Custom Search Engine
3. Add your search sites (or leave empty for entire web)
4. Get your `GOOGLE_CSE_ID` from the settings

### 3. Add to .env File

Edit your `.env` file and add:

```bash
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_search_engine_id_here
OPENAI_API_KEY=your_openai_key_here
```

### 4. Restart Chatbot

```bash
source venv/bin/activate
python 06_chatbot/app.py
```

You should see: `✅ Google Search tool enabled!`

## Usage Examples

**For Current Events:**

- "What's the latest news about AI?"
- "Who won the latest sports championship?"

**For Real-time Information:**

- "What's the current weather in New York?"
- "What's the price of Bitcoin today?"

**For Web Research:**

- "Find information about the newest machine learning papers"
- "What are the latest LangChain updates?"

## Pricing

- **Google Custom Search:** 100 free queries/day, then $5 per 1,000 queries
- **OpenAI API:** Based on model usage (gpt-4o-mini is cheap)

## Troubleshooting

**Problem:** "Google Search API keys not found in .env"

- Make sure you added `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` to `.env`
- Restart the chatbot after updating `.env`

**Problem:** Search returns no results

- Check that your Custom Search Engine includes the websites you want
- Verify your API key and CSE ID are correct

**Problem:** "Quota exceeded"

- You've exceeded the free 100 daily searches
- Upgrade to a paid plan or wait until tomorrow

## Alternative: Use DuckDuckGo (Free)

If you don't want to set up Google Search, you can use DuckDuckGo instead:

```python
@tool
def duckduckgo_search(query: str) -> str:
    """Search DuckDuckGo for information (free, no API key needed)."""
    from duckduckgo_search import DDGS
    try:
        results = DDGS().text(query, max_results=3)
        return str(results)
    except Exception as e:
        return f"Search failed: {str(e)}"
```

Then install: `pip install duckduckgo-search`

---

Happy searching! 🚀
