# §3.24–§3.31 portfolio — closeout summary

**Branch**: `multilingual_merged`  &nbsp;&nbsp;|&nbsp;&nbsp; **Last commit**: `0902381`  &nbsp;&nbsp;|&nbsp;&nbsp; **Period**: Apr 30 – May 1, 2026

A compact summary of the §3.24 → §3.31 optimisation pass on top of
the §3.21 baseline.  For the full chronological development journal
and every negative finding, see [`PROGRESS.md`](PROGRESS.md).

---

## What shipped (8 commits)

| Section | Commit | Nature | Net M3 Ultra | Net GGUF |
|--------:|:-------|:-------|:-------------|:---------|
| §3.24 | *(earlier)* | HiFT F16 conv kernels (64 tensors) | −3.6 ms HiFT | −33 MB |
| §3.25 | `c47c776` | FA flow-encoder — **negative finding**, reverted | docs | — |
| §3.26 | `daae187` | Missing `kernel_mul_mv_f32_f16{,_4,_short}` variants → 21 more HiFT F16 tensors | neutral | **−7.7 MB** |
| §3.27 | `52d184a` | `mul_mm + ADD(bias)[+residual]` fusion | neutral M3U (infra) | — |
| §3.28 | `64c991d` | + `GELU_ERF` fold-in (CFM FF ff0) | **−8.8 ms CFM** | — |
| §3.29 | `4633172` | Direct-store RMW — **negative finding**, reverted | docs | — |
| §3.30 | `145c822` | `test-metal-ops` fused-mul_mm harness + bias-only direct-store retry | neutral M3U (infra) | — |
| §3.31 | `0902381` | iOS-arm64 cross-build + `scripts/bench-m4-validation.sh` | infra | — |

**Net M3 Ultra**: CFM **541.9 → 534.0 ms (−7.9 ms / −1.5 %)**, S3Gen
**709 → 706 ms**, GGUF **754.4 → 746.7 MB (−7.7 MB)**.  Five
commits deliver measurable change; three are documented negative
findings or infrastructure work that de-risks future rounds.

## Parity guarantees

- **WAV byte-exact** across all 5 benched invocations on the shipping
  config (Q4_0 + HiFT F16 v2 GGUF, ES prompt, seed 42, `--temp 0
  --top-k 1 --n-gpu-layers 1`): md5 `d8a1b22375dbcb2259c686426a7d76c5`.
  Matches the §3.26 baseline exactly; §3.27/§3.28/§3.30 don't drift
  it by a single bit.
- **14 / 14 `test-metal-ops` gates PASS**:
  `diag_mask_inf`, `pad_ext`, 4× `conv_transpose_1d` (HiFT upsamples
  + tiny edge), 8× `mul_mm_fused` (covers CFM attn Q/K/V/out, FF
  gate/down, b=1, bc_out edge shapes, both bias and gelu fusion).
- **End-to-end smoke** across all 8 model pairs
  (2 T3 × 4 S3Gen variants): all produce correct output.
- **Streaming mode** (25-token chunks): 4 chunks, 938 ms first-chunk
  latency, no NaN/Inf.
- **Long-text** (309 tokens, 12.57 s audio): no NaN/Inf,
  speech-healthy RMS 1233.
- **Patch portability**:
  [`patches/ggml-metal-chatterbox-ops.patch`](patches/ggml-metal-chatterbox-ops.patch)
  (1088 lines) and `patches/ggml-opencl-chatterbox-ops.patch`
  (unmodified in this period) both apply cleanly via `git apply
  --check` on a fresh ggml clone at pinned `58c38058`.
- **iOS-arm64 cross-build**: `libggml-metal.a` + `libtts-cpp.a`
  compile clean for iOS 14.0+ arm64 with Xcode 16 / iOS 18.5 SDK —
  structural proof the §3.26/§3.27/§3.28/§3.30 kernel work is
  iOS-portable (no macOS-only intrinsics).

## Open follow-ups (tracked in PROGRESS)

| Item | Effort | Expected gain | Status |
|:-----|:-------|:--------------|:-------|
| M4 / iPhone / iPad validation of §3.24/§3.27/§3.28/§3.30 on bandwidth-limited silicon | 0.5–2 h on hardware | predicted +5–15 ms S3Gen; untested | hand-off script shipped (`scripts/bench-m4-validation.sh`); awaiting test host |
| Residual + gelu direct-store retry (with §3.30 harness as safety net) | 2–3 h | potential +3–8 ms M3 Ultra CFM | deferred; §3.29 negative finding root-caused to cooperative-store memory ordering, needs Metal memory-model audit |
| Extend fusion to other unary sub-ops (SILU / GELU / RELU / GELU_QUICK) | ~15 LOC each | 0 ms chatterbox (not in graph); useful downstream infra | deferred as pure-infra |
| Q4_0 HiFT via 2-D-on-disk storage + `conv1d_f32` branch | 1–2 days | +4–8 ms HiFT, −30 MB GGUF | deferred (large surgery: converter + C++) |
| T3 speculative decoding | 2–5 days | −130 to −200 ms T3 (−10 to −15 % wall) | largest remaining lever; needs its own planning session |

## Final bench — shipping config

`./build-metal/chatterbox --model models/chatterbox-t3-mtl-q4_0.gguf --s3gen-gguf models/chatterbox-s3gen-mtl-q4_0_hift_f16_v2.gguf --reference-audio /tmp/jfk.wav --text "Hola mundo, esta es una prueba multilingue." --language es --seed 42 --temp 0 --top-k 1 --n-gpu-layers 1 --out /tmp/cb.wav`

**M3 Ultra Metal, 5 invocations averaged:**

| Stage | Mean | Stdev |
|:------|-----:|------:|
| mel | 14.6 ms | 0.2 |
| `[encoder]` | 30.5 ms | 0.7 |
| `[cfm_total]` | **534.0 ms** | **1.3** |
| `[hift_decode]` | 121.1 ms | 0.6 |
| S3GEN_INFER_MS | **706.6 ms** | 4.5 |
| T3_INFER_MS | **432.6 ms** (84 tokens) | 2.2 |
| **Total inference** | **~1165 ms** | |
| **RTF** | **0.33** | |

Audio output: 3.48 s WAV from 84 speech tokens. Byte-exact and
deterministic.

## How to reproduce

```bash
# From the multilingual_merged branch HEAD
scripts/setup-ggml.sh                      # apply pinned ggml patches
cmake -S . -B build-metal -DGGML_METAL=ON -DGGML_BLAS=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j
./build-metal/test-metal-ops               # all 14 gates should PASS
bash scripts/bench-m4-validation.sh        # also works on M3 Ultra; prints Δ vs the reference baked into the script
```
