#pragma once

#include <string>
#include <vector>

namespace tts_cpp::supertonic {

struct EngineOptions {
    std::string model_gguf_path;
    std::string voice = "F1";
    std::string language = "en";
    int steps = 5;
    float speed = 1.05f;
    int seed = 42;
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

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
