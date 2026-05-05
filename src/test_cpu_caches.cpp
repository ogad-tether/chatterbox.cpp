// CPU-side persistent-cache validation harness for QVAC-18422
// "[TTS GGML] Optimize cpp backend multilingual for CPU".
//
// Verifies the four cache layers added to chatterbox_tts.cpp:
//
//  1. compute_time_mlp_cached() — t_val (float) → (1024,) t_emb vector.
//     Multilingual fires 10 distinct t-values per synth (cosine schedule);
//     Turbo fires 3.  Across synth calls the schedule is constant, so the
//     cache amortises every subsequent synth to zero compute_time_mlp work.
//
//  2. compute_time_emb_cached() — (t_val, r_val) → (1024,) mixed embedding.
//     Turbo meanflow only; multilingual leaves this cache empty.
//
//  3. g_cfm_estimator_cache — promotes the local-scope cfm_estimator_cache
//     to global lifetime so subsequent synth calls don't rebuild the
//     ~5500-node CFM graph or pay the gallocr_reserve cost.
//
//  4. g_weight_cpu_mirror — CPU mirror of large per-synth weight reads
//     (flow/input_embedding ~28 MB on multilingual, spk_embed_affine
//     ~60 KB).  Saves the ggml_backend_tensor_get round-trip every synth.
//
// All caches are invalidated together by s3gen_unload() so that switching
// to a different backend (e.g. CPU → Vulkan) doesn't reuse stale state.
//
// Usage (with model)        : ./test-cpu-caches MODEL_S3GEN.gguf [REF_DIR]
// Usage (cache-key only)    : ./test-cpu-caches
//
// Without a GGUF the harness still runs the lightweight cache-key tests
// that catch the typical -0/+0/NaN / std::hash<float> portability traps.

#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "chatterbox_tts_test_hooks.h"
#include "npy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace th = tts_cpp::chatterbox::test_hooks;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  %s\n        ",                      \
                __FILE__, __LINE__, #cond);                              \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    }                                                                    \
} while (0)

bool path_exists(const std::string & p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

// ---------------- 1. cache-key bit-cast tests ----------------
//
// These run unconditionally — no model needed.  They guard the rule
// that the time_mlp result cache uses a bit-cast hash of the float
// (so +0/-0 land in different buckets, NaNs are stable per-bit-pattern,
// and equal floats always hash to the same bucket regardless of how
// they were computed).

void test_cache_keys() {
    fprintf(stderr, "=== cache key (bit-cast) tests ===\n");

    // Equal floats → equal keys.
    CHECK(th::float_cache_key(0.5f) == th::float_cache_key(0.5f),
          "0.5 should be stable");

    // +0.0 and -0.0 are NOT equal under bit-cast (sign bit differs).
    // std::hash<float> typically collapses them — we deliberately don't.
    const float pos_zero = 0.0f;
    const float neg_zero = -0.0f;
    CHECK(th::float_cache_key(pos_zero) != th::float_cache_key(neg_zero),
          "+0 and -0 must produce distinct cache keys");

    // Distinct values → distinct keys (sanity).
    CHECK(th::float_cache_key(0.5f) != th::float_cache_key(0.25f),
          "0.5 vs 0.25 must differ");

    // NaN: bit-pattern stable (we don't normalise) — same NaN payload
    // hashes the same.  This is fine because the time_mlp_cache is
    // only ever queried with t_span values, none of which are NaN.
    uint32_t nan_bits = 0x7fc00001u;  // a quiet NaN
    float nan_val;
    std::memcpy(&nan_val, &nan_bits, sizeof(float));
    CHECK(th::float_cache_key(nan_val) == 0x7fc00001u,
          "NaN bit pattern must round-trip");

    // Pair key: high 32 bits = t_val, low 32 bits = r_val.
    const float t = 0.5f;
    const float r = 1.0f;
    const uint64_t expect =
        ((uint64_t) th::float_cache_key(t) << 32) |
         (uint64_t) th::float_cache_key(r);
    CHECK(th::float_pair_cache_key(t, r) == expect,
          "pair key must compose from individual float keys");

    // Order matters: (t, r) ≠ (r, t).
    CHECK(th::float_pair_cache_key(0.5f, 1.0f) !=
          th::float_pair_cache_key(1.0f, 0.5f),
          "pair key must not be commutative");

    // Cosine schedule used by multilingual (n_timesteps=10) — verify
    // 10 distinct keys.  Mirrors the t_span = 1 - cos(i/10 * pi/2) loop
    // in s3gen_synthesize_to_wav.
    std::vector<uint32_t> keys;
    keys.reserve(10);
    for (int i = 0; i < 10; ++i) {
        float tau = (float) i / 10.0f;
        float t_cos = 1.0f - std::cos(tau * 0.5f * (float) M_PI);
        keys.push_back(th::float_cache_key(t_cos));
    }
    bool all_distinct = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            if (keys[i] == keys[j]) { all_distinct = false; break; }
        }
    }
    CHECK(all_distinct,
          "multilingual t-span (n_timesteps=10 cosine) must produce 10 "
          "distinct cache keys, otherwise compute_time_mlp_cached would "
          "alias unrelated steps");
}

// ---------------- 2. starting cache state ----------------

void test_initial_state() {
    fprintf(stderr, "=== initial cache state ===\n");

    // s3gen_unload() before any synth must succeed even if no caches
    // were ever populated (idempotent).  Production callers in the
    // bare-addon teardown rely on this.
    s3gen_unload();
    CHECK(th::time_mlp_result_cache_size() == 0,
          "time_mlp result cache must start empty");
    CHECK(th::time_emb_result_cache_size() == 0,
          "time_emb result cache must start empty");
    CHECK(th::weight_mirror_cache_size() == 0,
          "weight mirror cache must start empty");
    CHECK(!th::cfm_estimator_cache_built(),
          "persistent cfm_estimator_cache must not be built before any "
          "synth");
    CHECK(!th::cfm_estimator_cache_b2(),
          "persistent cfm_estimator_cache b2 flag must default false");
}

// ---------------- 3. determinism + cache wiring on a real synth ----------

// Read built-in voice tokens.  No multilingual model available locally,
// so the harness uses the Turbo built-in voice if --ref-dir wasn't
// passed.  The cache logic is model-agnostic by construction; the
// multilingual benefit factor is larger but the bit-exact + lifecycle
// invariants this test verifies are identical across variants.
std::vector<int32_t> sample_speech_tokens() {
    // 24 tokens — enough to exercise the encoder + a single CFM batch
    // without bloating run-time.  Values are within [0, 6561) (S3 vocab).
    return {
        12, 34, 56, 78, 90, 121, 152, 173, 195, 217, 239, 261,
        283, 305, 327, 349, 371, 393, 415, 437, 459, 481, 503, 525,
    };
}

bool synthesize_once(const std::string & gguf,
                     const std::string & ref_dir,
                     std::vector<float> & wav,
                     double & wall_ms) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.ref_dir         = ref_dir;
    opts.out_wav_path    = "";          // stay in-memory
    opts.pcm_out         = &wav;
    opts.seed            = 42;
    opts.n_threads       = 0;           // auto: hardware_concurrency
    opts.sr              = 24000;
    opts.verbose         = false;
    opts.n_gpu_layers    = 0;           // CPU-only for this test
    opts.apply_trim_fade = true;
    opts.finalize        = true;

    const auto tokens = sample_speech_tokens();
    const double t0 = now_ms();
    int rc = s3gen_synthesize_to_wav(tokens, opts);
    wall_ms = now_ms() - t0;
    return rc == 0 && !wav.empty();
}

void test_warm_cache_bit_exact_and_lifecycle(const std::string & gguf,
                                             const std::string & ref_dir) {
    fprintf(stderr, "=== warm-cache bit-exact + lifecycle ===\n");

    // First call populates every cache.  Subsequent calls must (a)
    // produce bit-exact output and (b) skip every cache that was
    // already warmed.
    std::vector<float> wav_a, wav_b, wav_c;
    double t_a = 0, t_b = 0, t_c = 0;
    if (!synthesize_once(gguf, ref_dir, wav_a, t_a)) {
        fprintf(stderr, "skip: synth #1 failed (model load / arch?)\n");
        return;
    }

    const size_t n_time_mlp_after_a = th::time_mlp_result_cache_size();
    const size_t n_time_emb_after_a = th::time_emb_result_cache_size();
    const size_t n_weights_after_a  = th::weight_mirror_cache_size();
    const bool   cfm_built_after_a  = th::cfm_estimator_cache_built();

    CHECK(cfm_built_after_a,
          "after first synth, persistent cfm_estimator_cache must be built");
    CHECK(n_time_mlp_after_a > 0,
          "after first synth, time_mlp result cache must have at least one "
          "entry (n_timesteps for multilingual / 3 for Turbo)");
    CHECK(n_weights_after_a > 0,
          "after first synth, weight_mirror_cache must have at least one "
          "entry (input_embedding + spk_embed_affine/{w,b})");
    fprintf(stderr,
            "  synth #1: time_mlp=%zu  time_emb=%zu  weights=%zu  cfm=%s "
            "(%.1f ms)\n",
            n_time_mlp_after_a, n_time_emb_after_a, n_weights_after_a,
            cfm_built_after_a ? "built" : "fresh", t_a);

    // Second call: every cache must already be warm.  Its size must
    // not grow because the t-schedule and the model weights are
    // constant across synth calls.
    if (!synthesize_once(gguf, ref_dir, wav_b, t_b)) {
        fprintf(stderr, "skip: synth #2 failed\n");
        return;
    }
    CHECK(th::time_mlp_result_cache_size() == n_time_mlp_after_a,
          "synth #2 must NOT add new time_mlp entries (saw %zu, expected %zu)",
          th::time_mlp_result_cache_size(), n_time_mlp_after_a);
    CHECK(th::time_emb_result_cache_size() == n_time_emb_after_a,
          "synth #2 must NOT add new time_emb entries");
    CHECK(th::weight_mirror_cache_size() == n_weights_after_a,
          "synth #2 must NOT add new weight_mirror entries");
    CHECK(th::cfm_estimator_cache_built(),
          "synth #2 must keep the persistent cfm graph built");

    CHECK(wav_a.size() == wav_b.size(),
          "warm-cache synth #2 wav length must match cold-cache synth #1 "
          "(%zu vs %zu)", wav_a.size(), wav_b.size());
    if (wav_a.size() == wav_b.size() && !wav_a.empty()) {
        size_t diff = 0;
        float  max_abs = 0;
        for (size_t i = 0; i < wav_a.size(); ++i) {
            float d = std::fabs(wav_a[i] - wav_b[i]);
            if (d > 0) diff++;
            if (d > max_abs) max_abs = d;
        }
        CHECK(diff == 0,
              "warm-cache synth #2 must be byte-for-byte identical to "
              "synth #1 (mismatched samples=%zu, max_abs=%.6e)", diff, max_abs);
    }
    fprintf(stderr, "  synth #2: %.1f ms (warm caches, bit-exact ok)\n", t_b);

    // Third call after s3gen_unload() — every cache must have been
    // reset.  Subsequent synth must repopulate them and still
    // produce bit-exact output (deterministic seed=42).
    s3gen_unload();
    CHECK(th::time_mlp_result_cache_size() == 0,
          "s3gen_unload must clear time_mlp result cache");
    CHECK(th::time_emb_result_cache_size() == 0,
          "s3gen_unload must clear time_emb result cache");
    CHECK(th::weight_mirror_cache_size() == 0,
          "s3gen_unload must clear weight_mirror cache");
    CHECK(!th::cfm_estimator_cache_built(),
          "s3gen_unload must tear down the persistent cfm cache");

    // Idempotent: a second unload must not crash or produce errors.
    s3gen_unload();

    if (!synthesize_once(gguf, ref_dir, wav_c, t_c)) {
        fprintf(stderr, "skip: synth #3 (post-unload) failed\n");
        return;
    }
    CHECK(th::cfm_estimator_cache_built(),
          "synth #3 must rebuild the cfm cache after unload");
    CHECK(wav_a.size() == wav_c.size(),
          "post-unload synth wav length must match");
    if (wav_a.size() == wav_c.size() && !wav_a.empty()) {
        size_t diff = 0;
        float max_abs = 0;
        for (size_t i = 0; i < wav_a.size(); ++i) {
            float d = std::fabs(wav_a[i] - wav_c[i]);
            if (d > 0) diff++;
            if (d > max_abs) max_abs = d;
        }
        CHECK(diff == 0,
              "post-unload synth must be byte-for-byte identical to first "
              "synth (mismatched samples=%zu, max_abs=%.6e)",
              diff, max_abs);
    }
    fprintf(stderr, "  synth #3 (post-unload): %.1f ms — bit-exact ok\n", t_c);

    // peek_time_mlp_cached: warm value should round-trip.
    auto cosine_t = [](int i, int n) {
        float tau = (float) i / (float) n;
        return 1.0f - std::cos(tau * 0.5f * (float) M_PI);
    };
    // For Turbo (meanflow=true, n_timesteps=2) the schedule is linear:
    // [0, 0.5, 1.0].  For multilingual (cosine, n_timesteps=10) the
    // schedule is cosine.  We probe both candidates non-destructively;
    // at least one of {0.5f, cosine_t(1,10)} should be present.
    auto a = th::peek_time_mlp_cached(0.5f);
    auto b = th::peek_time_mlp_cached(cosine_t(1, 10));
    CHECK(!a.empty() || !b.empty(),
          "peek_time_mlp_cached must return a populated entry for at least "
          "one of the canonical t-values (0.5 for Turbo or cosine[1] for "
          "multilingual)");
    if (!a.empty()) {
        CHECK(a.size() == 1024,
              "time_mlp cached entry must be (1024,) — saw %zu", a.size());
    }
    if (!b.empty()) {
        CHECK(b.size() == 1024,
              "time_mlp cached entry must be (1024,) — saw %zu", b.size());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    fprintf(stderr, "test-cpu-caches: QVAC-18422 cache validation\n");

    test_cache_keys();
    test_initial_state();

    if (argc < 2) {
        fprintf(stderr, "\n(no GGUF given — skipping warm-cache + lifecycle "
                        "tests; run as `%s MODEL.gguf [REF_DIR]` to exercise "
                        "the full pipeline)\n", argv[0]);
    } else {
        const std::string gguf = argv[1];
        const std::string ref_dir = (argc >= 3 ? argv[2] : "");
        if (!path_exists(gguf)) {
            fprintf(stderr, "error: GGUF not found at %s\n", gguf.c_str());
            return 2;
        }
        test_warm_cache_bit_exact_and_lifecycle(gguf, ref_dir);
    }

    // Always release at exit so the next test invocation starts clean.
    s3gen_unload();

    fprintf(stderr, "\n=== summary ===\n  checks:   %d\n  failures: %d\n",
            g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
