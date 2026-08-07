/* Host verification of the on-device RAG pipeline: load the QA model + tokenizer and
 * a story into buffers (exactly what the ESP32 build does), then retrieve + answer.
 * Auto-asks the "<role> named <Name>" questions like rag_poc/rag_demo.py, so the
 * output can be diffed against the Python demo for the same checkpoint and story.
 *
 *   gcc -O2 -o host_rag_test host_rag_test.c -lm
 *   ./host_rag_test <model_q80.bin> <tok.bin> <story.txt> [top_k]
 */
#include "rag_core.h"

static uint8_t* slurp(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((sz + 4) & ~3L);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    buf[sz] = '\0';
    if (out_size) *out_size = sz;
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model tok story.txt [top_k]\n", argv[0]); return 1; }
    int top_k = argc > 4 ? atoi(argv[4]) : 2;

    uint8_t* model_buf = slurp(argv[1], NULL);
    uint8_t* tok_buf = slurp(argv[2], NULL);
    long story_len;
    char* story = (char*)slurp(argv[3], &story_len);

    Transformer t;
    if (!llama2_load(&t, model_buf, 0)) { fprintf(stderr, "load failed\n"); return 1; }
    Tokenizer tok;
    build_tokenizer(&tok, tok_buf, t.config.vocab_size);

    char* work = (char*)malloc(story_len + 4);   // mutable copy (rag_split rewrites it)
    char ans[128], ctx[RAG_CTX_BUF];

    // scan for "<role> named <Name>" (mirrors rag_demo.py's regex) and ask each
    const char* p = story;
    while ((p = strstr(p, " named ")) != NULL) {
        const char* re = p;
        const char* rs = re; while (rs > story && rs[-1] >= 'a' && rs[-1] <= 'z') rs--;
        const char* q = p + 7;
        if (rs < re && q[0] >= 'A' && q[0] <= 'Z') {
            const char* ne = q + 1; while (*ne >= 'a' && *ne <= 'z') ne++;
            char role[32], gold[32], question[64];
            int rl = (int)(re - rs); if (rl > 31) rl = 31;
            memcpy(role, rs, rl); role[rl] = '\0';
            int nl = (int)(ne - q); if (nl > 31) nl = 31;
            memcpy(gold, q, nl); gold[nl] = '\0';
            snprintf(question, sizeof(question), "What was the %s's name?", role);

            memcpy(work, story, story_len + 1);
            rag_ask(&t, &tok, work, question, top_k, 10, ans, sizeof(ans), ctx, sizeof(ctx));
            printf("Q: %s\n  retrieved: %s\n  A: %s   gold: %s  ->  %s\n\n",
                   question, ctx, ans, gold, strcmp(ans, gold) == 0 ? "OK" : "MISS");
        }
        p += 7;
    }
    return 0;
}
