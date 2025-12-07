# RAG Context Window Management

Four techniques to stop your RAG chatbot from degrading over long conversations.

## Run It

**Option A: Local (requires Node.js 18+)**
```bash
git clone <repo>
cd rag-context-window
npm install
export OPENAI_API_KEY='sk-...'   # optional, only for semantic cache
npm run demo
```

**Option B: Docker (no dependencies)**
```bash
git clone <repo>
cd rag-context-window
docker build -t rag-context .
docker run -e OPENAI_API_KEY='sk-...' rag-context
```

## What Each Solution Does

| Solution | Problem | Fix |
|----------|---------|-----|
| **1. Conversation Buffer** | Bot "forgets" things mentioned earlier as context fills up | Compress old messages into summaries, extract critical entities (account #s, locations) |
| **2. Document Ordering** | Retrieved docs get ignored even when relevant | Reorder docs so best ones are at start/end (where LLMs pay most attention) |
| **3. Semantic Cache** | Paying to answer "refund policy?" and "how to get money back?" separately | Cache responses by meaning, not exact text match |
| **4. Multi-Agent** | Complex cross-domain questions get shallow answers | Split query across specialized sub-agents, then synthesize |

## Run Individual Demos

```bash
npm run demo:buffer      # conversation compression
npm run demo:ordering    # document reordering (no API calls)
npm run demo:cache       # semantic caching (needs OPENAI_API_KEY)
npm run demo:multiagent  # multi-agent orchestration
```

## API Keys
- **OpenAI**: https://platform.openai.com → only needed for semantic cache (Solution 3)
