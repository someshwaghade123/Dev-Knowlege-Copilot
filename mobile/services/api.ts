// mobile/services/api.ts
// ─────────────────────────────────────────────────────────────────────────────
// All HTTP calls to the FastAPI backend live here.
// The mobile app NEVER constructs URLs elsewhere — only imports from this file.
//
// WHY CENTRALISE API CALLS?
//   If the backend URL changes (e.g. from localhost to production) we only
//   update BASE_URL in one place. The rest of the app is unaffected.
//
// RESPONSE CONTRACT (matches backend/api/routes.py QueryResponse):
//   {
//     answer:     string
//     citations:  { title: string, source_url: string|null, text_preview: string }[]
//     confidence: "high" | "medium" | "low"
//     latency_ms: number
//     tokens_used: number
//   }

// ── CONFIGURATION ─────────────────────────────────────────────────────────────
// BASE_URL is the most common point of failure in mobile development.
// 
// 1. ANDROID EMULATOR: Use "http://10.0.2.2:8000/api/v1"
// 2. REAL DEVICE (iOS/Android): Use your PC's local IP (e.g. "http://192.168.1.5:8000/api/v1")
//    - Find your IP by running `ipconfig` in your PC terminal.
//    - Your phone and PC MUST be on the same Wi-Fi.
//    - You MUST start the backend with `uvicorn ... --host 0.0.0.0`
//
const BASE_URL = "http://10.53.167.156:8000/api/v1";

/**
 * Enhanced fetch with a timeout to prevent infinite "loading" states.
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 10000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(id);
        return response;
    } catch (err: any) {
        clearTimeout(id);
        if (err.name === "AbortError") {
            throw new Error("Request timed out (Backend unreachable). Verify your BASE_URL and IP.");
        }
        throw err;
    }
}

export interface Citation {
    title: string;
    source_url: string | null;
    text_preview: string;
}

export interface QueryResponse {
    answer: string;
    citations: Citation[];
    confidence: "high" | "medium" | "low";
    latency_ms: number;
    tokens_used: number;
}

export interface HealthResponse {
    status: string;
    indexed_vectors: number;
    env: string;
}

/**
 * POST /api/v1/query
 * Sends a user question to the RAG backend and returns the structured response.
 */
export async function queryDocuments(
    query: string,
    topK: number = 5
): Promise<QueryResponse> {
    const response = await fetchWithTimeout(`${BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail ?? `Request failed: ${response.status}`);
    }

    return response.json() as Promise<QueryResponse>;
}

/**
 * GET /api/v1/health
 * Quick liveness check — useful to show connection status in the UI.
 */
export async function checkHealth(): Promise<HealthResponse> {
    const response = await fetchWithTimeout(`${BASE_URL}/health`);
    if (!response.ok) throw new Error("Backend unreachable");
    return response.json() as Promise<HealthResponse>;
}

/**
 * GET /api/v1/documents
 * Fetches the list of all indexed documents.
 */
export async function listDocuments(): Promise<{ count: number; documents: any[] }> {
    const response = await fetchWithTimeout(`${BASE_URL}/documents`);
    if (!response.ok) throw new Error("Failed to fetch documents");
    return response.json();
}
