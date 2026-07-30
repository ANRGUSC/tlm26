/* Portable int8 Llama-2 inference core for TinyStories sub-1M models.
 *
 * This is Karpathy's runq.c forward pass, unchanged in its math, with the three
 * host-only dependencies removed so it compiles for an ESP32 as well as a PC:
 *   - weights are read from an in-memory buffer instead of an mmap'd file
 *   - the tokenizer is read from an in-memory buffer instead of a file
 *   - generated text is delivered through an emit() callback instead of printf
 * Nothing in the numerics changed, so a host build of this core reproduces
 * runq.exe byte-for-byte (see host_test.c); that is what makes the ESP32 build
 * trustworthy without a device in hand.
 */
#ifndef LLAMA2_CORE_H
#define LLAMA2_CORE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>

/* Allocation hook: the ESP32 build routes large buffers to PSRAM by defining
 * L2_MALLOC before including this header. The host build uses plain malloc. */
#ifndef L2_MALLOC
#define L2_MALLOC malloc
#endif
#ifndef L2_FREE
#define L2_FREE free
#endif

static int GS = 0; // quantization group size, set from the checkpoint header

// ---------------------------------------------------------------------------
// model structs (identical layout to runq.c)

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct { int8_t* q; float* s; } QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    QuantizedTensor *wq, *wk, *wv, *wo, *w1, *w2, *w3;
    float* rms_final_weight;
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2;
    QuantizedTensor xq, hq;
    float *q, *k, *v, *att, *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

// ---------------------------------------------------------------------------
// run state

static int l2_malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)L2_MALLOC(p->dim * sizeof(float));
    s->xb = (float*)L2_MALLOC(p->dim * sizeof(float));
    s->xb2 = (float*)L2_MALLOC(p->dim * sizeof(float));
    s->hb = (float*)L2_MALLOC(p->hidden_dim * sizeof(float));
    s->hb2 = (float*)L2_MALLOC(p->hidden_dim * sizeof(float));
    s->xq.q = (int8_t*)L2_MALLOC(p->dim * sizeof(int8_t));
    s->xq.s = (float*)L2_MALLOC(p->dim * sizeof(float));
    s->hq.q = (int8_t*)L2_MALLOC(p->hidden_dim * sizeof(int8_t));
    s->hq.s = (float*)L2_MALLOC(p->hidden_dim * sizeof(float));
    s->q = (float*)L2_MALLOC(p->dim * sizeof(float));
    s->k = (float*)L2_MALLOC(kv_dim * sizeof(float));
    s->v = (float*)L2_MALLOC(kv_dim * sizeof(float));
    s->att = (float*)L2_MALLOC(p->n_heads * p->seq_len * sizeof(float));
    s->logits = (float*)L2_MALLOC(p->vocab_size * sizeof(float));
    s->key_cache = (float*)L2_MALLOC(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = (float*)L2_MALLOC(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    return s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q && s->k && s->v
        && s->att && s->logits && s->key_cache && s->value_cache
        && s->xq.q && s->xq.s && s->hq.q && s->hq.s;
}

// ---------------------------------------------------------------------------
// quantization

static void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = qx->q[i] * qx->s[i / GS];
}

static void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            qx->q[group * GS + i] = (int8_t) roundf(quant_value);
        }
    }
}

static QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor*)L2_MALLOC(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p;
    return res;
}

static void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    float* fptr = (float*) ptr;
    w->rms_att_weight = fptr; fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr; fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr; fptr += p->dim;
    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // The full dequantized embedding table would cost vocab*dim*4 bytes of RAM
    // (131 KB at 279K, 262 KB at 989K). On a microcontroller with no PSRAM that is
    // the difference between fitting and not, so it is dequantized one token-row at
    // a time inside forward() instead. Numerically identical either way.
    w->token_embedding_table = NULL;
    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

/* Parse a v2 (int8) checkpoint that already lives in memory. `data` must point at
 * the 4-byte magic and stay resident for the model's lifetime, because the weight
 * pointers index directly into it (on an ESP32 this is a flash address).
 *
 * max_context, if > 0 and smaller than the trained seq_len, shrinks the kv-cache
 * allocation to that many positions. The kv-cache dominates RAM, so this is what
 * lets the model fit internal SRAM on a microcontroller with no PSRAM. Positions
 * 0..max_context-1 are in the RoPE range the model was trained on, so generation
 * up to that length is unaffected. Returns 1 on success. */
static int llama2_load(Transformer* t, const uint8_t* data, int max_context) {
    uint32_t magic; memcpy(&magic, data, 4);
    if (magic != 0x616b3432u) return 0;             // "ak42"
    int version; memcpy(&version, data + 4, 4);
    if (version != 2) return 0;
    memcpy(&t->config, data + 8, sizeof(Config));   // 7 ints
    if (max_context > 0 && max_context < t->config.seq_len) t->config.seq_len = max_context;
    uint8_t shared_classifier = data[8 + sizeof(Config)];
    int group_size; memcpy(&group_size, data + 8 + sizeof(Config) + 1, 4);
    GS = group_size;
    void* weights_ptr = (void*)(data + 256);        // v2 header is 256 bytes
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_classifier);
    return l2_malloc_run_state(&t->state, &t->config);
}

// ---------------------------------------------------------------------------
// transformer forward (identical to runq.c, OpenMP pragmas dropped)

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++)
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

static float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // dequantize just this token's embedding row (see memory_map_weights)
    for (int i = 0; i < dim; i++)
        x[i] = w->q_tokens->q[token*dim + i] * w->q_tokens->s[(token*dim + i) / GS];

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i], v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                att[t] = score / sqrtf(head_size);
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ---------------------------------------------------------------------------
// tokenizer (reads from a buffer instead of a file)

typedef struct { char *str; int id; } TokenIndex;
typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

/* Build from the tok*.bin layout: [max_token_length:int][ (score:float, len:int, bytes) * vocab ]. */
static void build_tokenizer(Tokenizer* t, const uint8_t* data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)L2_MALLOC(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)L2_MALLOC(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    const uint8_t* p = data;
    memcpy(&t->max_token_length, p, 4); p += 4;
    for (int i = 0; i < vocab_size; i++) {
        memcpy(t->vocab_scores + i, p, 4); p += 4;
        int len; memcpy(&len, p, 4); p += 4;
        t->vocab[i] = (char*)L2_MALLOC(len + 1);
        memcpy(t->vocab[i], p, len); p += len;
        t->vocab[i][len] = '\0';
    }
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { str, 0 };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

static void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)L2_MALLOC(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* str_buffer = (char*)L2_MALLOC((t->max_token_length*2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    L2_FREE(str_buffer);
}

// ---------------------------------------------------------------------------
// sampler (identical to runq.c)

typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature, topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(float* p, int n) {
    int max_i = 0; float max_p = p[0];
    for (int i = 1; i < n; i++) if (p[i] > max_p) { max_i = i; max_p = p[i]; }
    return max_i;
}
static int sample_mult(float* p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
static int compare_probindex(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a; ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}
static int sample_topp(float* p, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (p[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = p[i]; n0++; }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);
    float cumulative_prob = 0.0f; int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) { cumulative_prob += probindex[i].prob; if (cumulative_prob > topp) { last_idx = i; break; } }
    float r = coin * cumulative_prob, cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; }
    return probindex[last_idx].index;
}
static void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size; s->temperature = temperature; s->topp = topp; s->rng_state = seed;
    s->probindex = (ProbIndex*)L2_MALLOC(vocab_size * sizeof(ProbIndex));
}
static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
static int sample(Sampler* s, float* logits) {
    int next;
    if (s->temperature == 0.0f) {
        next = sample_argmax(logits, s->vocab_size);
    } else {
        for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
        softmax(logits, s->vocab_size);
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0 || s->topp >= 1) next = sample_mult(logits, s->vocab_size, coin);
        else next = sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
    }
    return next;
}

// ---------------------------------------------------------------------------
// generation loop; emits decoded pieces through a caller callback

static void generate(Transformer* t, Tokenizer* tok, Sampler* sampler,
                     const char* prompt, int steps, void (*emit)(const char*)) {
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)L2_MALLOC((strlen(prompt) + 3) * sizeof(int));
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(t, token, pos);
        int next;
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
        else next = sample(sampler, logits);
        pos++;
        if (next == 1) break; // BOS ends a story
        char* piece = decode(tok, token, next);
        if (piece && piece[0] != '\0' && !(piece[1] == '\0' &&
            !(isprint((unsigned char)piece[0]) || isspace((unsigned char)piece[0]))))
            emit(piece);
        token = next;
    }
    L2_FREE(prompt_tokens);
}

#endif // LLAMA2_CORE_H
