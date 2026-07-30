/* Host verification: load the model and tokenizer into memory buffers (exactly
 * what the ESP32 build does) and generate, so the output can be diffed against
 * runq.exe. If they match byte-for-byte, the buffer-based port in llama2_core.h
 * is numerically identical to the reference engine.
 *
 *   gcc -O2 -o host_test host_test.c -lm
 *   ./host_test <model_q80.bin> <tok.bin> <temp> <seed> <steps> "<prompt>"
 */
#include "llama2_core.h"

static uint8_t* slurp(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    // malloc returns >=8-byte-aligned memory, enough for the float* views into
    // the weight buffer (the ESP32 build gets the same alignment from ps_malloc)
    uint8_t* buf = (uint8_t*)malloc((sz + 3) & ~3L);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static void emit_stdout(const char* s) { fputs(s, stdout); fflush(stdout); }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s model tok temp seed steps prompt\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    const char* tok_path = argv[2];
    float temperature = atof(argv[3]);
    unsigned long long seed = strtoull(argv[4], NULL, 10);
    int steps = atoi(argv[5]);
    const char* prompt = argv[6];

    uint8_t* model_buf = slurp(model_path, NULL);
    uint8_t* tok_buf = slurp(tok_path, NULL);

    Transformer t;
    if (!llama2_load(&t, model_buf, 0)) { fprintf(stderr, "load failed\n"); return 1; }

    Tokenizer tok;
    build_tokenizer(&tok, tok_buf, t.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, t.config.vocab_size, temperature, 0.9f, seed);

    if (steps <= 0 || steps > t.config.seq_len) steps = t.config.seq_len;

    // generate() re-emits the prompt tokens as it decodes them, exactly as runq.c
    // does, so the prompt is not echoed separately here.
    generate(&t, &tok, &sampler, prompt, steps, emit_stdout);
    printf("\n");
    return 0;
}
