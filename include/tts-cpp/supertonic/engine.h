#pragma once

#include "tts-cpp/export.h"

#include <string>
#include <vector>

namespace tts_cpp::supertonic {

struct EngineOptions {
    std::string model_gguf_path;
    // Empty / zero values use the defaults stored in the GGUF metadata.
    std::string voice;
    std::string language = "en";
    int steps = 0;
    float speed = 0.0f;
    int seed = 42;
    int n_threads = 0;
    int n_gpu_layers = 0;
    // Optional path to a .npy file containing the initial noise tensor of shape
    // [1, latent_channels, latent_len] (float32).  When provided, latent_len is
    // taken from the npy file (overriding the duration-predicted length) and
    // the seeded RNG is bypassed.  Useful for byte-exact reproduction of an
    // ONNX/PyTorch reference run.
    std::string noise_npy_path;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int sample_rate = 44100;
    float duration_s = 0.0f;
};

TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
