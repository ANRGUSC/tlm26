/* Host verification of the MoE engine (llama2_moe_core.h): load an MoE int8 blob +
 * tokenizer into buffers (exactly what the ESP32 build does) and generate.
 *   gcc -O2 -o host_moe_test host_moe_test.c -lm
 *   ./host_moe_test <moe.bin> <tok.bin> <temp> <seed> <steps> "<prompt>"
 */
#include "llama2_moe_core.h"

static uint8_t* slurp(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((sz + 3) & ~3L);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static void emit_stdout(const char* s) { fputs(s, stdout); fflush(stdout); }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s moe tok temp seed steps prompt\n", argv[0]); return 1; }
    uint8_t* model_buf = slurp(argv[1], NULL);
    uint8_t* tok_buf = slurp(argv[2], NULL);
    float temperature = atof(argv[3]);
    unsigned long long seed = strtoull(argv[4], NULL, 10);
    int steps = atoi(argv[5]);
    const char* prompt = argv[6];

    Transformer t;
    if (!llama2_load(&t, model_buf, 0)) { fprintf(stderr, "load failed\n"); return 1; }
    fprintf(stderr, "loaded: dim=%d L=%d E=%d topk=%d vocab=%d seq=%d\n",
            t.config.dim, t.config.n_layers, t.config.n_experts, t.config.moe_top_k,
            t.config.vocab_size, t.config.seq_len);

    Tokenizer tok;
    build_tokenizer(&tok, tok_buf, t.config.vocab_size);
    Sampler sampler;
    build_sampler(&sampler, t.config.vocab_size, temperature, 0.9f, seed);
    if (steps <= 0 || steps > t.config.seq_len) steps = t.config.seq_len;

    generate(&t, &tok, &sampler, prompt, steps, emit_stdout);
    printf("\n");
    return 0;
}
