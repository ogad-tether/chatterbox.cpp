#pragma once

// Back half of the Chatterbox pipeline: S3Gen encoder → 2-step meanflow CFM →
// HiFT vocoder. Takes T3-generated speech tokens + reference voice conditioning
// and writes a 24 kHz WAV.
//
// Implementation in src/s3gen_pipeline.cpp.

#include <cstdint>
#include <string>
#include <vector>

struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;  // required: chatterbox-s3gen.gguf
    std::string out_wav_path;     // required: where to write the 24 kHz wav

    // If empty, use the built-in voice embedded in the GGUF
    // (s3gen/builtin/{embedding,prompt_token,prompt_feat}).
    // Otherwise load embedding.npy / prompt_token.npy / prompt_feat.npy from
    // this directory.
    std::string ref_dir;

    int  seed      = 42;
    int  n_threads = 0;          // 0 = hardware_concurrency
    int  sr        = 24000;
    bool debug     = false;      // validation mode; requires ref_dir
};

// Runs encoder + CFM + HiFT on the given T3 speech tokens and writes a WAV.
// Returns 0 on success, non-zero on error.
int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts);
