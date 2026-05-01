#include "supertonic_internal.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <stdexcept>

namespace tts_cpp::supertonic::detail {
namespace {

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, require_key(ctx, key));
}

float get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, require_key(ctx, key));
}

bool get_bool_u32(const gguf_context * ctx, const char * key, bool fallback = false) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(ctx, id) != 0;
}

std::string get_string(const gguf_context * ctx, const char * key, const std::string & fallback = {}) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_str(ctx, id);
}

std::vector<std::string> get_string_array(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    size_t n = gguf_get_arr_n(ctx, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return out;
}

ggml_tensor * get_tensor_or_null(const supertonic_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    return it == model.tensors.end() ? nullptr : it->second;
}

} // namespace

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name) {
    ggml_tensor * t = get_tensor_or_null(model, name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name) {
    auto it = model.source_tensors.find(source_name);
    if (it == model.source_tensors.end() || !it->second) {
        throw std::runtime_error("missing source tensor: " + source_name);
    }
    return it->second;
}

bool load_supertonic_gguf(const std::string & path, supertonic_model & model) {
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gp);
    if (!gguf_ctx) {
        fprintf(stderr, "load_supertonic_gguf: failed to open '%s'\n", path.c_str());
        return false;
    }

    try {
        std::string arch = get_string(gguf_ctx, "supertonic.arch");
        if (arch != "supertonic2" && arch != "supertonic") {
            throw std::runtime_error("unexpected supertonic.arch: " + arch);
        }

        model.hparams.sample_rate = (int) get_u32(gguf_ctx, "supertonic.sample_rate");
        model.hparams.base_chunk_size = (int) get_u32(gguf_ctx, "supertonic.base_chunk_size");
        model.hparams.ttl_chunk_compress_factor =
            (int) get_u32(gguf_ctx, "supertonic.ttl_chunk_compress_factor");
        model.hparams.latent_dim = (int) get_u32(gguf_ctx, "supertonic.latent_dim");
        model.hparams.latent_channels = (int) get_u32(gguf_ctx, "supertonic.latent_channels");
        model.hparams.default_steps = (int) get_u32(gguf_ctx, "supertonic.default_steps");
        model.hparams.default_speed = get_f32(gguf_ctx, "supertonic.default_speed");
        model.hparams.language_wrap = get_bool_u32(gguf_ctx, "supertonic.language_wrap", true);
        model.hparams.default_voice = get_string(gguf_ctx, "supertonic.default_voice", "F1");
        model.languages = get_string_array(gguf_ctx, "supertonic.languages");
        model.tts_json = get_string(gguf_ctx, "supertonic.tts_json");

        model.backend = ggml_backend_cpu_init();
        if (!model.backend) throw std::runtime_error("ggml_backend_cpu_init failed");

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params params = {
            /*.mem_size=*/ ggml_tensor_overhead() * (size_t) num_tensors,
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc=*/ true,
        };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init failed");

        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            if (!src) throw std::runtime_error(std::string("missing tmp tensor: ") + name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        if (!model.buffer_w) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w);
             cur;
             cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }

        {
            ggml_tensor * unicode = require_tensor(model, "supertonic/unicode_indexer");
            model.unicode_indexer.resize((size_t) ggml_nelements(unicode));
            ggml_backend_tensor_get(unicode, model.unicode_indexer.data(), 0, ggml_nbytes(unicode));
        }

        std::vector<std::string> tensor_names = get_string_array(gguf_ctx, "supertonic.tensor_names");
        std::vector<std::string> source_names = get_string_array(gguf_ctx, "supertonic.source_names");
        if (tensor_names.size() != source_names.size()) {
            throw std::runtime_error("supertonic tensor/source metadata length mismatch");
        }
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            ggml_tensor * t = require_tensor(model, tensor_names[i]);
            model.source_tensors[source_names[i]] = t;
        }

        for (const std::string & voice_name : get_string_array(gguf_ctx, "supertonic.voice_names")) {
            supertonic_voice_style voice;
            voice.name = voice_name;
            voice.ttl = require_tensor(model, "supertonic/voices/" + voice_name + "/ttl");
            voice.dp  = require_tensor(model, "supertonic/voices/" + voice_name + "/dp");
            model.voices[voice_name] = voice;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "load_supertonic_gguf: %s\n", e.what());
        gguf_free(gguf_ctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        free_supertonic_model(model);
        return false;
    }

    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}

void free_supertonic_model(supertonic_model & model) {
    if (model.buffer_w) {
        ggml_backend_buffer_free(model.buffer_w);
        model.buffer_w = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
    }
    model.tensors.clear();
    model.source_tensors.clear();
    model.voices.clear();
    model.unicode_indexer.clear();
    model.languages.clear();
    model.tts_json.clear();
}

} // namespace tts_cpp::supertonic::detail
