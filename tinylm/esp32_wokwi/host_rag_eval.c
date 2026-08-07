/* Batch end-to-end accuracy of the on-device (int8) RAG pipeline, mirroring
 * rag_poc/eval_e2e.py: over N unseen stories, ask each "<role> named <Name>"
 * question (once per distinct name; role-typed only when the story has >=2 names,
 * as in the Python eval), retrieve + answer, tally exact match. Lets us compare the
 * quantized C engine's accuracy against the fp32 PyTorch reference (0.988 at 989K).
 *
 *   gcc -O2 -o host_rag_eval host_rag_eval.c -lm
 *   ./host_rag_eval <model_q80.bin> <tok.bin> <valid.txt> [n_stories] [top_k]
 */
#include "rag_core.h"

static uint8_t* slurp(const char* p, long* n) {
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* b = (uint8_t*)malloc(s + 4); fread(b, 1, s, f); fclose(f); b[s] = '\0';
    if (n) *n = s; return b;
}

static int is_role_ok(const char* r) {
    static const char* OK[] = {"boy","girl","man","woman","dog","cat","bird","bunny",
        "rabbit","bear","fish","duck","frog","mouse","monkey","elephant","lion","tiger",
        "fox","squirrel","turtle","puppy","kitten","horse","cow","pig","sheep","chicken",
        "dragon","dinosaur","robot","princess","prince","king","queen","baby","brother",
        "sister","friend","butterfly","bee","ant","owl","snake"};
    for (unsigned i = 0; i < sizeof(OK)/sizeof(OK[0]); i++) if (!strcmp(r, OK[i])) return 1;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model tok valid.txt [n_stories] [top_k]\n", argv[0]); return 1; }
    int n_stories = argc > 4 ? atoi(argv[4]) : 400;
    int top_k = argc > 5 ? atoi(argv[5]) : 2;

    uint8_t* mb = slurp(argv[1], NULL); uint8_t* tb = slurp(argv[2], NULL);
    char* text = (char*)slurp(argv[3], NULL);
    Transformer t; if (!llama2_load(&t, mb, 0)) { fprintf(stderr, "load failed\n"); return 1; }
    Tokenizer tok; build_tokenizer(&tok, tb, t.config.vocab_size);

    char work[8192], ans[128];
    int tot = 0, hit = 0, recall = 0, done = 0;
    char* p = text;   // walk the file, splitting on the literal "<|endoftext|>"
    while (done < n_stories && p && *p) {
        char* sep = strstr(p, "<|endoftext|>");
        int slen = sep ? (int)(sep - p) : (int)strlen(p);
        // trim
        char* s = p; int L = slen;
        while (L > 0 && (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) { s++; L--; }
        while (L > 0 && (s[L-1]==' '||s[L-1]=='\n'||s[L-1]=='\r'||s[L-1]=='\t')) L--;
        if (L > 0 && L < (int)sizeof(work) - 1) {
            char story_buf[8192];
            memcpy(story_buf, s, L); story_buf[L] = '\0';
            done++;
            // collect distinct names (first role + gold-name substring for recall)
            char names[16][32], roles[16][32]; int nn = 0;
            for (char* q = strstr(story_buf, " named "); q; q = strstr(q + 7, " named ")) {
                char* re = q; char* rs = re; while (rs > story_buf && rs[-1]>='a'&&rs[-1]<='z') rs--;
                char* nm = q + 7;
                if (rs < re && nm[0]>='A'&&nm[0]<='Z') {
                    char* ne = nm + 1; while (*ne>='a'&&*ne<='z') ne++;
                    char nmv[32]; int nl=(int)(ne-nm); if(nl>31)nl=31; memcpy(nmv,nm,nl); nmv[nl]='\0';
                    int seen=0; for(int k=0;k<nn;k++) if(!strcmp(names[k],nmv)) seen=1;
                    if(!seen && nn<16){ int rl=(int)(re-rs); if(rl>31)rl=31;
                        memcpy(roles[nn],rs,rl); roles[nn][rl]='\0'; strcpy(names[nn],nmv); nn++; }
                }
            }
            int multi = nn >= 2;
            for (int k = 0; k < nn; k++) {
                if (multi && !is_role_ok(roles[k])) continue;
                char question[64];
                snprintf(question, sizeof(question), "What was the %s's name?", roles[k]);
                memcpy(work, story_buf, L + 1);
                char ctx[RAG_CTX_BUF];
                rag_ask(&t, &tok, work, question, top_k, 10, ans, sizeof(ans), ctx, sizeof(ctx));
                int ok = !strcmp(ans, names[k]);
                tot++; hit += ok;
                if (strstr(ctx, names[k])) recall++;   // gold name present in retrieved ctx
            }
        }
        if (!sep) break;
        p = sep + strlen("<|endoftext|>");
    }
    printf("ckpt=%s  stories=%d  questions=%d\n", argv[1], done, tot);
    printf("  end-to-end EM    = %d/%d = %.3f\n", hit, tot, tot ? (double)hit/tot : 0);
    printf("  ctx-has-gold     = %d/%d = %.3f\n", recall, tot, tot ? (double)recall/tot : 0);
    return 0;
}
