/**
 * RAG Context Window Management
 * 4 solutions for production RAG systems
 *
 * Run: npm install && npm run demo
 * Requires: OPENAI_API_KEY
 */

import OpenAI from "openai";
import { createHash } from "crypto";

// ============================================================
// SOLUTION 1: Conversation Summary Buffer
// Compresses older messages while preserving critical entities
// ============================================================

interface Message { role: "user" | "assistant"; content: string }

class ConversationSummaryBuffer {
  private client = new OpenAI();
  private messages: Message[] = [];
  private summary = "";
  private entities = { ids: new Set<string>(), locations: new Set<string>(), products: new Set<string>() };

  constructor(private maxRecent = 6) {}

  async addMessage(role: "user" | "assistant", content: string) {
    this.messages.push({ role, content });
    // Extract entities before any compression
    content.match(/(?:account|order|ticket)[\s#:]*(\d{6,})/gi)?.forEach(id => {
      const num = id.match(/\d{6,}/)?.[0];
      if (num) this.entities.ids.add(num);
    });
    content.match(/(?:in|from|shipping to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/g)?.forEach(loc => {
      this.entities.locations.add(loc.replace(/^(in|from|shipping to)\s+/i, ""));
    });
    content.match(/\b(Pro|Enterprise|Basic|Premium)\b/gi)?.forEach(p => this.entities.products.add(p.toLowerCase()));

    if (this.messages.length > this.maxRecent) await this.compress();
  }

  private async compress() {
    const old = this.messages.slice(0, -this.maxRecent);
    this.messages = this.messages.slice(-this.maxRecent);
    const text = old.map(m => `${m.role}: ${m.content}`).join("\n");
    const res = await this.client.chat.completions.create({
      model: "gpt-4o-mini", max_tokens: 500,
      messages: [{ role: "user", content: `Summarize concisely:\n${text}\nPrevious: ${this.summary || "None"}` }],
    });
    this.summary = res.choices[0].message.content || "";
  }

  buildContext(docs: string[]): Message[] {
    const parts = ["You are a helpful assistant."];
    if (this.summary) parts.push(`\n\nHistory:\n${this.summary}`);
    const active = Object.entries(this.entities).filter(([, s]) => s.size > 0);
    if (active.length) parts.push(`\n\nKey info:\n${active.map(([k, s]) => `- ${k}: ${[...s].join(", ")}`).join("\n")}`);
    if (docs.length) parts.push(`\n\nDocs:\n${docs.join("\n\n---\n\n")}`);
    return [{ role: "user", content: parts.join("") }, { role: "assistant", content: "I understand." }, ...this.messages];
  }

  getState() { return { summary: this.summary, entities: Object.fromEntries(Object.entries(this.entities).map(([k, v]) => [k, [...v]])), recent: this.messages.length }; }
}

// ============================================================
// SOLUTION 2: Strategic Document Ordering
// Best docs at start/end to combat U-shaped attention
// ============================================================

interface ScoredDoc { content: string; score: number }

function reorderForAttention(docs: ScoredDoc[]): string[] {
  const sorted = [...docs].sort((a, b) => b.score - a.score);
  if (sorted.length <= 2) return sorted.map(d => d.content);
  const result: ScoredDoc[] = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    i % 2 === 1 ? result.push(sorted[i]) : result.splice(Math.floor(result.length / 2), 0, sorted[i]);
  }
  return result.map(d => d.content);
}

// ============================================================
// SOLUTION 3: Semantic Caching (in-memory for simplicity)
// Cache by meaning, not exact match
// ============================================================

class SemanticCache {
  private openai = new OpenAI();
  private cache = new Map<string, { response: string; embedding: number[] }>();
  private stats = { hits: 0, misses: 0 };

  constructor(private threshold = 0.92) {}

  private async embed(text: string) {
    const res = await this.openai.embeddings.create({ model: "text-embedding-3-small", input: text });
    return res.data[0].embedding;
  }

  private cosine(a: number[], b: number[]) {
    const dot = a.reduce((s, v, i) => s + v * b[i], 0);
    return dot / (Math.sqrt(a.reduce((s, v) => s + v * v, 0)) * Math.sqrt(b.reduce((s, v) => s + v * v, 0)));
  }

  async get(query: string): Promise<{ response: string; similarity: number } | null> {
    const emb = await this.embed(query);
    let best = { score: 0, response: "" };
    for (const c of this.cache.values()) {
      const sim = this.cosine(emb, c.embedding);
      if (sim > best.score && sim >= this.threshold) best = { score: sim, response: c.response };
    }
    if (best.response) { this.stats.hits++; return { response: best.response, similarity: best.score }; }
    this.stats.misses++;
    return null;
  }

  async set(query: string, response: string) {
    this.cache.set(createHash("sha256").update(query).digest("hex"), { response, embedding: await this.embed(query) });
  }

  getStats() { return { ...this.stats, hitRate: this.stats.hits / (this.stats.hits + this.stats.misses) || 0 }; }
}

// ============================================================
// SOLUTION 4: Multi-Agent Architecture
// Decompose complex queries across specialized sub-agents
// ============================================================

type Retriever = (query: string) => Promise<string[]>;

class MultiAgentRAG {
  private client = new OpenAI();

  async orchestrate(query: string, retrievers: Record<string, Retriever>) {
    // Plan
    const planRes = await this.client.chat.completions.create({
      model: "gpt-4o", max_tokens: 500,
      messages: [{ role: "user", content: `Break down for retrieval.\nQuery: ${query}\nSources: ${Object.keys(retrievers).join(", ")}\nRespond JSON: {"subtasks": [{"source": "...", "subquery": "..."}]}` }],
    });
    const planText = planRes.choices[0].message.content || "{}";
    const plan = JSON.parse(planText.match(/\{[\s\S]*\}/)?.[0] || '{"subtasks":[]}');

    // Execute in parallel
    const results = await Promise.all(plan.subtasks.map(async (t: { source: string; subquery: string }) => {
      const docs = await retrievers[t.source](t.subquery);
      const res = await this.client.chat.completions.create({
        model: "gpt-4o-mini", max_tokens: 300,
        messages: [{ role: "user", content: `Answer: ${t.subquery}\n\nDocs:\n${docs.join("\n\n")}` }],
      });
      return { source: t.source, findings: res.choices[0].message.content || "" };
    }));

    // Synthesize
    const synthRes = await this.client.chat.completions.create({
      model: "gpt-4o", max_tokens: 1000,
      messages: [{ role: "user", content: `Question: ${query}\n\nFindings:\n${results.map(r => `**${r.source}**:\n${r.findings}`).join("\n\n")}\n\nSynthesize a comprehensive answer.` }],
    });
    return { answer: synthRes.choices[0].message.content || "", plan, results };
  }
}

// ============================================================
// DEMOS
// ============================================================

const div = () => console.log("\n" + "=".repeat(50) + "\n");

async function demoBuffer() {
  console.log("SOLUTION 1: Conversation Summary Buffer\n");
  const buf = new ConversationSummaryBuffer(4);
  const conv = [
    ["user", "Hi, issues with account #123456"], ["assistant", "I'll help with account #123456."],
    ["user", "I'm in Canada, Pro subscription shipping broken"], ["assistant", "Checking Pro subscription for Canada."],
    ["user", "Order #789012 never arrived"], ["assistant", "Looking into order #789012."],
    ["user", "Shipping to Toronto"], ["assistant", "Got it - Toronto."],
  ] as const;
  for (const [role, content] of conv) { await buf.addMessage(role, content); process.stdout.write("."); }
  console.log("\n\nState:", JSON.stringify(buf.getState(), null, 2));
}

function demoOrdering() {
  console.log("SOLUTION 2: Document Ordering\n");
  const docs: ScoredDoc[] = [
    { content: "Doc A", score: 0.95 }, { content: "Doc B", score: 0.88 },
    { content: "Doc C", score: 0.72 }, { content: "Doc D", score: 0.65 }, { content: "Doc E", score: 0.82 },
  ];
  console.log("Original:", docs.map(d => `${d.content}(${d.score})`).join(" > "));
  const reordered = reorderForAttention(docs);
  console.log("Reordered:", reordered.map(c => { const d = docs.find(x => x.content === c)!; return `${c}(${d.score})`; }).join(" > "));
}

async function demoCache() {
  console.log("SOLUTION 3: Semantic Caching\n");
  const cache = new SemanticCache(0.85);
  await cache.set("What is your refund policy?", "Returns within 30 days for full refund.");
  for (const q of ["How do I get my money back?", "Can I return for refund?", "What's the weather?"]) {
    const hit = await cache.get(q);
    console.log(`"${q}" -> ${hit ? `HIT (${hit.similarity.toFixed(2)})` : "MISS"}`);
  }
  console.log("Stats:", cache.getStats());
}

async function demoMultiAgent() {
  console.log("SOLUTION 4: Multi-Agent RAG\n");
  const retrievers = {
    pricing: async () => ["Enterprise: $99/mo", "Pro: $49/mo", "Basic: $19/mo"],
    competitors: async () => ["Competitor A: $120/mo", "Competitor B: $79/mo"],
    feedback: async () => ["92% satisfaction", "NPS +15 after Q3"],
  };
  const { answer, plan } = await new MultiAgentRAG().orchestrate(
    "Compare enterprise pricing to competitors and summarize feedback", retrievers
  );
  console.log("Plan:", plan.subtasks.map((t: { source: string }) => t.source).join(", "));
  console.log("\nAnswer:", answer);
}

// Main
const arg = process.argv[2];
if (!process.env.OPENAI_API_KEY) { console.error("Set OPENAI_API_KEY"); process.exit(1); }

(async () => {
  try {
    if (!arg) { await demoBuffer(); div(); demoOrdering(); div(); await demoCache(); div(); await demoMultiAgent(); }
    else if (arg === "buffer") await demoBuffer();
    else if (arg === "ordering") demoOrdering();
    else if (arg === "cache") await demoCache();
    else if (arg === "multiagent") await demoMultiAgent();
    else { console.error("Options: buffer, ordering, cache, multiagent"); process.exit(1); }
    console.log("\n✓ Done");
  } catch (e) { console.error(e); process.exit(1); }
})();
