# 06 — Interview Q&A for Week 1

These are the 15 most likely questions you will be asked about your Week 1 implementation.
Each answer is **concise, specific, and backed by your real code**.

---

## Section 1: System Design

**Q1: Walk me through your RAG pipeline from when a user sends a query to when they get a response.**

> "The user's query hits `POST /api/v1/query` in FastAPI. First, we embed the query using BGE-small-en-v1.5 with an instruction prefix — this gives us a 384-dimensional float32 vector. We then search FAISS IndexFlatIP for the top-5 closest chunk vectors, which returns faiss_ids and cosine similarity scores (~40ms total). We query SQLite for the full chunk text and source metadata for those IDs. We then build a prompt: a system message instructing the LLM to answer only from context, and a user message containing the 5 chunks plus the question. We call the LLM API asynchronously and await the response (~400ms). Finally, we assemble a JSON response containing the answer, citation list, confidence label, total latency, and token count."

---

**Q2: Why did you choose RAG over fine-tuning an LLM?**

> "RAG is practical for this use case for three reasons. First, our document corpus changes — new docs get indexed regularly, and fine-tuning would require retraining every time. Second, RAG provides citations — we can show the user exactly which document the answer came from, making it auditable. Third, fine-tuning is expensive and complex; RAG gave us a working system in Week 1 using a free open-source model."

---

**Q3: What would happen at scale — say, 1 million documents?**

> "Three things change. First, FAISS: I'd replace IndexFlatIP (exact, O(N)) with IndexIVFFlat (approximate, ~10x faster with <5% recall loss) or HNSW for even lower latency. Second, ingestion: I'd run embedding in parallel batches on GPU instead of CPU. Third, SQLite: I'd migrate to PostgreSQL for concurrent writes and better query performance. The FastAPI layer itself is stateless and horizontally scalable — just add more instances behind a load balancer."

---

## Section 2: Technical Decisions

**Q4: Why 384-token chunks? Why not 512 (the model's maximum)?**

> "BGE-small has a 512-token limit, but embedding 512 tokens right at the boundary risks silent truncation on some inputs. I chose 384 to leave 128 tokens of headroom. I also compared 256/384/512 on a benchmark query set and measured precision@3 — 384 gave the best balance of specificity and context richness. Smaller chunks (256) were too sparse; larger chunks (512) retrieved correctly but included too much irrelevant context."

---

**Q5: Why do you use overlap between chunks?**

> "To prevent context loss at chunk boundaries. A sentence that starts at the end of chunk N and finishes at the start of chunk N+1 would be split, and neither chunk represents it fully. With 64-token overlap, that sentence appears complete in at least one chunk, making it retrievable. The tradeoff is ~20% more chunks, which is worth it for the retrieval quality improvement."

---

**Q6: You use Inner Product (IP) in FAISS, not L2. Why?**

> "Both work for similarity search, but they're equivalent when vectors are L2-normalised. Our embedder normalises all vectors to unit length (`normalize_embeddings=True`). For unit vectors: cosine_similarity(A, B) = dot_product(A, B). So IndexFlatIP with normalised vectors gives exact cosine similarity — which is the right metric for text embeddings. L2 distance on normalised vectors gives the same ranking anyway, so it's purely a computational choice: IP is slightly more direct."

---

**Q7: How do you store the link between FAISS search results and actual document metadata?**

> "FAISS returns integer IDs (positions in the index). I use a `faiss_id` column in the SQLite `chunks` table to bridge them. When I add embeddings to FAISS, I track the sequential IDs it assigns (0, 1, 2, ...) and store them with the chunk text and document metadata in SQLite. At query time: FAISS returns IDs → SQLite SELECT WHERE faiss_id IN (...) → chunk text + title + URL."

---

## Section 3: Architecture

**Q8: Why store metadata in SQLite instead of FAISS?**

> "FAISS is optimised for one thing: fast vector arithmetic. It has no concept of metadata, filtering, or structured queries. SQLite handles those perfectly — filtering by source URL, sorting by ingestion date, joining documents to their chunks. This separation of concerns means each component does what it's best at. If we needed metadata filtering at search time (e.g. 'only search FastAPI docs'), we'd add an inverted index or use Chroma which has built-in metadata filtering."

---

**Q9: How does your backend handle the case where the user asks something your docs don't cover?**

> "Two layers. First, if FAISS scores are all below 0.60, we label confidence as 'low' — the mobile app shows a warning badge. Second, the LLM system prompt explicitly instructs: 'If the answer is not in the context, say I could not find a reliable answer.' We track how often this phrase appears in responses — that's our Week 5 hallucination heuristic. We don't trust the LLM to stay in scope on its own; we engineer constraints into the prompt."

---

**Q10: Why FastAPI over Flask or Django?**

> "For this specific system, FastAPI is the right tool because: (1) The LLM call is async I/O — FastAPI handles concurrent requests during that 400ms wait natively; (2) Pydantic models validate request/response schemas automatically; (3) Auto-generated /docs endpoint is useful during development. Flask works too but requires manual async handling. Django is overkill — we don't need an ORM or admin panel."

---

## Section 4: Production Thinking

**Q11: What are your biggest latency bottlenecks and how would you address them?**

> "The LLM call is ~400ms, 85% of total latency. Fix: Redis cache on query hash — identical questions return instantly (Week 6 plan). Second is CPU embedding at ~60ms — fix is GPU or a smaller model like all-MiniLM (22M params) if we need sub-20ms. FAISS search at ~10ms is negligible for our scale. I've instrumented each stage separately so I know exactly where time goes."

---

**Q12: How do you make the system fault-tolerant?**

> "Three patterns I've implemented or planned: (1) LLM timeout with `httpx.AsyncClient(timeout=30.0)` — we don't wait forever; (2) Empty index check — returns 503 immediately rather than crashing; (3) Score threshold filter — if no chunk exceeds min_score, we return a graceful 'no relevant results' response instead of an LLM call. For production, I'd add exponential backoff retry on LLM API failures."

---

**Q13: How would you evaluate the quality of your RAG system?**

> "Two approaches. Offline: I built a golden test set of 20 query-answer pairs. For each query I check if the correct document appears in top-3 retrieved chunks — that's precision@3. Online: I log every query, the confidence score, and the LLM response. If I see 'I could not find an answer' frequently for reasonable queries, retrieval quality is poor. I also track latency P50/P95 in the request_logs table."

---

**Q14: What is the 'cold start' problem and how does it affect your system?**

> "Cold start happens when the FAISS index isn't loaded into memory yet. On a fresh container or restart, `load_or_create()` reads the binary index from disk — this takes 1–5 seconds for a 20k-vector index. During that window, the app returns 503. Solutions: (1) Kubernetes readiness probe (don't route traffic until healthy); (2) Pre-warm by keeping replica running on restart; (3) Bake a pre-loaded index into the Docker image for zero cold start."

---

**Q15: Walk me through what happens when I run `ingest_docs.py`.**

> "The script reads all .md and .txt files from the specified folder. For each file, it chunks the text into 384-token overlapping windows using tiktoken. All chunks for a document are embedded in a single batch call to BGE-small — this is efficient because sentence-transformers processes them in parallel. The resulting float32 embeddings are added to FAISS (which assigns sequential IDs). Each chunk's text, token count, and the assigned FAISS ID are saved to SQLite. At the end, the FAISS index is serialised to disk as a binary file. Total time for 100 documents: roughly 30–60 seconds on CPU."
