"""Emit base64 PROGMEM headers for the model and tokenizer blobs, for the Wokwi
ESP32 sketch. Base64 keeps the source ~2.7x smaller than a hex byte array, and the
sketch decodes each blob into PSRAM at boot. Run from the esp32_wokwi directory."""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BLOBS = [
    # (symbol, path, header file)
    ("model_279k", "out_qat_int4_dep_long/model_q80.bin", "model_279k.h"),
    ("tok_279k",   "data/tok512dep.bin",                  "tok_279k.h"),
    ("model_989k", "out_width_d128_long/model_q80.bin",   "model_989k.h"),
    ("tok_989k",   "data/tok512.bin",                     "tok_989k.h"),
]


def emit_raw(symbol, path, header):
    """Emit the blob as a 4-byte-aligned PROGMEM byte array. On the ESP32 flash is
    memory-mapped, so the model's weight pointers can index straight into this array
    with no RAM copy and no PSRAM. Larger source than base64, but zero runtime RAM."""
    with open(os.path.join(ROOT, path), "rb") as f:
        raw = f.read()
    guard = symbol.upper() + "_RAW_H"
    parts = []
    for i in range(0, len(raw), 20):
        parts.append("  " + ",".join(str(b) for b in raw[i:i + 20]) + ",")
    body = "\n".join(parts)
    with open(os.path.join(HERE, header), "w") as f:
        f.write(f"// Auto-generated from {path} ({len(raw)} bytes). Do not edit.\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#ifndef PROGMEM\n#define PROGMEM\n#endif\n\n")
        f.write(f"static const unsigned int {symbol}_len = {len(raw)};\n")
        f.write(f"__attribute__((aligned(4))) static const unsigned char {symbol}[] PROGMEM = {{\n{body}\n}};\n\n")
        f.write(f"#endif // {guard}\n")
    print(f"{header}: raw={len(raw):,}  source~{os.path.getsize(os.path.join(HERE, header)):,} bytes")


def emit(symbol, path, header):
    with open(os.path.join(ROOT, path), "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    # split into ~120-char lines inside a single PROGMEM string literal
    lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]
    body = "\n".join(f'  "{ln}"' for ln in lines)
    guard = symbol.upper() + "_H"
    with open(os.path.join(HERE, header), "w") as f:
        f.write(f"// Auto-generated from {path} ({len(raw)} bytes). Do not edit.\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"static const unsigned int {symbol}_raw_len = {len(raw)};\n")
        f.write(f"static const char {symbol}_b64[] PROGMEM =\n{body};\n\n")
        f.write(f"#endif // {guard}\n")
    print(f"{header}: raw={len(raw):,}  b64={len(b64):,}  source~{os.path.getsize(os.path.join(HERE, header)):,} bytes")


for sym, path, hdr in BLOBS:
    emit(sym, path, hdr)

# raw flash-resident arrays for the no-PSRAM build (weights + tokenizer)
print("--- raw (flash-resident, no PSRAM) ---")
emit_raw("model_279k", "out_qat_int4_dep_long/model_q80.bin", "model_279k_raw.h")
emit_raw("tok_279k",   "data/tok512dep.bin",                  "tok_279k_raw.h")
emit_raw("model_989k", "out_width_d128_long/model_q80.bin",   "model_989k_raw.h")
emit_raw("tok_989k",   "data/tok512.bin",                     "tok_989k_raw.h")
