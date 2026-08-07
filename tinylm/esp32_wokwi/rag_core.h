/* On-device RAG for the tiny QA model: retrieve -> prompt -> answer, in portable C.
 *
 * The retriever is a byte-for-byte port of rag_poc/rag_demo.py: lowercase [a-z']+
 * words, a tiny stemmer (strip possessive, fold named->name), TF-IDF overlap between
 * the question and each sentence chunk, top-k by score. No embeddings, no index, no
 * float tables beyond the log() — it runs in a few KB on an ESP32. The retrieved
 * chunks plus the question are formatted exactly like the fine-tuning data and the
 * QA model answers greedily (argmax), so a host build reproduces the Python demo.
 *
 * Depends on llama2_core.h for the model/tokenizer (include it first).
 */
#ifndef RAG_CORE_H
#define RAG_CORE_H

#include "llama2_core.h"

#ifndef RAG_MAX_SENTS
#define RAG_MAX_SENTS 64
#endif
#ifndef RAG_MAX_WORD
#define RAG_MAX_WORD 32
#endif
#ifndef RAG_MAX_QWORDS
#define RAG_MAX_QWORDS 32
#endif
#ifndef RAG_CTX_BUF
#define RAG_CTX_BUF 2048
#endif

/* --- tiny stemmer + stopwords, identical to rag_demo.py norm()/STOP --- */
static void rag_norm(char* w) {
    int L = (int)strlen(w);
    if (L >= 2 && w[L - 1] == 's' && w[L - 2] == '\'') { w[L - 2] = '\0'; }  // strip 's
    if (strcmp(w, "named") == 0) strcpy(w, "name");
}

static int rag_is_stop(const char* w) {
    static const char* STOP[] = {"the", "a", "an", "was", "is", "were", "what", "who",
        "of", "to", "and", "in", "it", "he", "she", "they", "his", "her", "s"};
    for (unsigned i = 0; i < sizeof(STOP) / sizeof(STOP[0]); i++)
        if (strcmp(w, STOP[i]) == 0) return 1;
    return 0;
}

/* Count occurrences of normalized word `w` in `text` (lowercased [a-z']+ tokens). */
static int rag_word_count(const char* text, const char* w) {
    int cnt = 0, bi = 0;
    char buf[RAG_MAX_WORD];
    for (const char* c = text;; c++) {
        char ch = *c;
        int isw = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '\'';
        if (isw) {
            if (bi < RAG_MAX_WORD - 1)
                buf[bi++] = (ch >= 'A' && ch <= 'Z') ? (char)(ch - 'A' + 'a') : ch;
        } else {
            if (bi > 0) {
                buf[bi] = '\0';
                rag_norm(buf);
                if (buf[0] && strcmp(buf, w) == 0) cnt++;
                bi = 0;
            }
            if (ch == '\0') break;
        }
    }
    return cnt;
}

/* Split `story` into sentence chunks: boundary = whitespace preceded by . ! ? " —
 * mirrors rag_demo.py's re.split(r'(?<=[.!?"])\s+'). Newlines are treated as spaces.
 * Returns the number of chunks; each chunk is a NUL-terminated slice of `work`. */
static int rag_split(char* work, char* sents[], int max_sents) {
    for (char* c = work; *c; c++) if (*c == '\n' || *c == '\r' || *c == '\t') *c = ' ';
    int len = (int)strlen(work), i = 0, n = 0;
    while (i < len && n < max_sents) {
        while (i < len && work[i] == ' ') i++;      // skip leading spaces
        int start = i;
        while (i < len) {                            // to next boundary or end
            if (work[i] == ' ' && i > start &&
                (work[i - 1] == '.' || work[i - 1] == '!' ||
                 work[i - 1] == '?' || work[i - 1] == '"')) break;
            i++;
        }
        int end = i;
        while (end > start && work[end - 1] == ' ') end--;   // trim trailing spaces
        if (end > start) {
            work[end] = '\0';                        // terminate this chunk in place
            sents[n++] = work + start;
        }
        i++;                                          // step past the split space
    }
    return n;
}

/* Retrieve top-k chunks for `question`, join in document order into `out`.
 * Same scoring and tie-break as rag_demo.py: score = sum_w cb[w]*log(1+N/df[w]);
 * ties broken by higher index first (Python sorts (score,index) descending). */
static void rag_retrieve(char* sents[], int n, const char* question,
                         int top_k, char* out, int out_sz) {
    char qwords[RAG_MAX_QWORDS][RAG_MAX_WORD];
    int nq = 0, bi = 0;
    char buf[RAG_MAX_WORD];
    for (const char* c = question;; c++) {           // unique, normed, non-stop q-words
        char ch = *c;
        int isw = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '\'';
        if (isw) {
            if (bi < RAG_MAX_WORD - 1)
                buf[bi++] = (ch >= 'A' && ch <= 'Z') ? (char)(ch - 'A' + 'a') : ch;
        } else {
            if (bi > 0) {
                buf[bi] = '\0'; rag_norm(buf);
                if (buf[0] && !rag_is_stop(buf)) {
                    int seen = 0;
                    for (int q = 0; q < nq; q++) if (strcmp(qwords[q], buf) == 0) seen = 1;
                    if (!seen && nq < RAG_MAX_QWORDS) strcpy(qwords[nq++], buf);
                }
                bi = 0;
            }
            if (ch == '\0') break;
        }
    }

    float score[RAG_MAX_SENTS];
    int order[RAG_MAX_SENTS];
    for (int i = 0; i < n; i++) { score[i] = 0.0f; order[i] = i; }
    for (int q = 0; q < nq; q++) {
        int df = 0;
        for (int i = 0; i < n; i++) if (rag_word_count(sents[i], qwords[q]) > 0) df++;
        if (df == 0) continue;
        float idf = logf(1.0f + (float)n / (float)df);
        for (int i = 0; i < n; i++) {
            int c = rag_word_count(sents[i], qwords[q]);
            if (c > 0) score[i] += c * idf;
        }
    }
    // selection sort by (score desc, index desc) — matches Python's reverse tuple sort
    for (int a = 0; a < n; a++) {
        int best = a;
        for (int b = a + 1; b < n; b++) {
            if (score[order[b]] > score[order[best]] ||
                (score[order[b]] == score[order[best]] && order[b] > order[best]))
                best = b;
        }
        int t = order[a]; order[a] = order[best]; order[best] = t;
    }
    int chosen[RAG_MAX_SENTS], nc = 0;
    for (int a = 0; a < n && a < top_k; a++) if (score[order[a]] > 0.0f) chosen[nc++] = order[a];
    if (nc == 0 && n > 0) chosen[nc++] = 0;
    for (int a = 0; a < nc; a++)                     // sort chosen ascending (doc order)
        for (int b = a + 1; b < nc; b++)
            if (chosen[b] < chosen[a]) { int t = chosen[a]; chosen[a] = chosen[b]; chosen[b] = t; }

    out[0] = '\0';
    int oi = 0;
    for (int a = 0; a < nc; a++) {
        if (a > 0 && oi < out_sz - 1) out[oi++] = ' ';
        const char* s = sents[chosen[a]];
        for (int k = 0; s[k] && oi < out_sz - 1; k++) out[oi++] = s[k];
    }
    out[oi] = '\0';
}

/* Greedy (argmax) answer to a fully-formed prompt; stops at EOS/BOS/newline or
 * max_new tokens. Writes the first answer line, whitespace-trimmed, to `out`.
 * Mirrors rag_poc/eval_qa.py answer(): temp 0, decode, split("\n")[0].strip(). */
static void rag_answer(Transformer* t, Tokenizer* tok, const char* prompt,
                       int max_new, char* out, int out_sz) {
    int* toks = (int*)L2_MALLOC((strlen(prompt) + 3) * sizeof(int));
    int n = 0;
    encode(tok, prompt, 1, 0, toks, &n);
    float* logits = NULL;
    for (int pos = 0; pos < n; pos++) logits = forward(t, toks[pos], pos);  // consume prompt
    int prev = toks[n - 1];
    int oi = 0;
    for (int g = 0; g < max_new; g++) {
        int next = sample_argmax(logits, t->config.vocab_size);
        if (next == 1 || next == 2) break;                                  // BOS/EOS
        char* piece = decode(tok, prev, next);
        for (int k = 0; piece[k]; k++) {
            if (piece[k] == '\n') { g = max_new; break; }                   // stop at newline
            if (oi < out_sz - 1) out[oi++] = piece[k];
        }
        prev = next;
        if (n + g >= t->config.seq_len - 1) break;
        logits = forward(t, next, n + g);
    }
    out[oi] = '\0';
    // strip leading/trailing whitespace
    int s = 0; while (out[s] == ' ' || out[s] == '\t') s++;
    int e = (int)strlen(out); while (e > s && (out[e - 1] == ' ' || out[e - 1] == '\t')) e--;
    memmove(out, out + s, e - s); out[e - s] = '\0';
    L2_FREE(toks);
}

/* Full pipeline: retrieve over `story`, build the prompt, answer. `ctx_out` (optional,
 * may be NULL) receives the retrieved context for display. */
static void rag_ask(Transformer* t, Tokenizer* tok, char* story_work,
                    const char* question, int top_k, int max_new,
                    char* ans_out, int ans_sz, char* ctx_out, int ctx_sz) {
    char* sents[RAG_MAX_SENTS];
    int n = rag_split(story_work, sents, RAG_MAX_SENTS);
    char ctx[RAG_CTX_BUF];
    rag_retrieve(sents, n, question, top_k, ctx, sizeof(ctx));
    if (ctx_out) { strncpy(ctx_out, ctx, ctx_sz - 1); ctx_out[ctx_sz - 1] = '\0'; }
    char prompt[RAG_CTX_BUF + 256];
    snprintf(prompt, sizeof(prompt), "%s\nQuestion: %s\nAnswer:", ctx, question);
    rag_answer(t, tok, prompt, max_new, ans_out, ans_sz);
}

#endif // RAG_CORE_H
