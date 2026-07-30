/* TinyStories sub-1M language model on an ESP32-S3, for the Wokwi online simulator.
 *
 * No PSRAM required. The int8 weights live in flash (a PROGMEM byte array) and are
 * read in place, because ESP32 flash is memory-mapped; nothing copies them into RAM.
 * Each token's embedding row is dequantized on the fly rather than held as a table,
 * so the only real RAM cost is the kv-cache, which is capped by MAX_CONTEXT. That is
 * how a ~300 KB model runs inside a bare ESP32-S3's internal SRAM.
 *
 * The forward pass is Karpathy's runq.c, ported in llama2_core.h and verified
 * byte-identical to the desktop engine. Type a story beginning in the Serial
 * Monitor and the model completes it.
 *
 * --- Switch models here -------------------------------------------------------
 *   279 = QAT-int4 deployable, 279K params (default; fits the Wokwi web IDE)
 *   989 = width d128, 989K params, best quality (3.8 MB of source; use the Wokwi
 *         VS Code extension, and note ~1 MB of weights needs a large-app partition)
 * The 989K weights (~1 MB) exceed a bare ESP32-S3's 512 KB SRAM only if copied to
 * RAM; here they stay in flash, so the constraint is flash/app-partition size. */
#define USE_MODEL 279

#define MAX_CONTEXT 128     // kv-cache positions kept in SRAM (<= trained seq_len 256)
#define MAX_NEW_TOKENS 120
#define TEMPERATURE 0.8f
#define TOPP 0.9f
// -----------------------------------------------------------------------------

#define L2_MALLOC malloc    // internal SRAM; no PSRAM
#define L2_FREE   free
#include "llama2_core.h"

#if USE_MODEL == 279
  #include "model_279k_raw.h"
  #include "tok_279k_raw.h"
  #define MODEL_DATA model_279k
  #define MODEL_LEN  model_279k_len
  #define TOK_DATA   tok_279k
  #define MODEL_NAME "QAT-int4 deployable (279K)"
#else
  #include "model_989k_raw.h"
  #include "tok_989k_raw.h"
  #define MODEL_DATA model_989k
  #define MODEL_LEN  model_989k_len
  #define TOK_DATA   tok_989k
  #define MODEL_NAME "width d128 (989K)"
#endif

static Transformer g_model;
static Tokenizer   g_tok;
static Sampler     g_sampler;

static void emit_serial(const char* s) { Serial.print(s); }

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== TinyStories on ESP32-S3 (no PSRAM) ===");
  Serial.printf("model: %s   int8 weights in flash: %u bytes\n", MODEL_NAME, MODEL_LEN);
  Serial.printf("free heap before load: %u KB\n", (unsigned)(ESP.getFreeHeap() / 1024));

  // weights are read straight from flash; only the kv-cache + activations use RAM
  if (!llama2_load(&g_model, (const uint8_t*)MODEL_DATA, MAX_CONTEXT)) {
    Serial.println("ERROR: bad checkpoint or out of memory for the kv-cache.");
    Serial.println("Lower MAX_CONTEXT and retry.");
    return;
  }
  build_tokenizer(&g_tok, (const uint8_t*)TOK_DATA, g_model.config.vocab_size);
  build_sampler(&g_sampler, g_model.config.vocab_size, TEMPERATURE, TOPP, 1337ULL);

  Serial.printf("ready. dim=%d layers=%d vocab=%d context=%d\n",
                g_model.config.dim, g_model.config.n_layers,
                g_model.config.vocab_size, g_model.config.seq_len);
  Serial.printf("free heap after load: %u KB\n", (unsigned)(ESP.getFreeHeap() / 1024));
  Serial.println("\nType a story beginning and press Enter:");
}

void loop() {
  static char line[256];
  static int len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len == 0) continue;
      line[len] = '\0';
      Serial.printf("\n> %s", line);
      int steps = MAX_NEW_TOKENS;
      if (steps > g_model.config.seq_len) steps = g_model.config.seq_len;
      unsigned long t0 = millis();
      generate(&g_model, &g_tok, &g_sampler, line, steps, emit_serial);
      unsigned long dt = millis() - t0;
      Serial.printf("\n[%lu ms]\n\nNext beginning:\n", dt);
      len = 0;
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}
