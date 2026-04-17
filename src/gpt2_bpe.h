#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gpt2_bpe {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string>                 id_to_token;
    std::unordered_map<std::string, int>     bpe_ranks;

    bool load_vocab_json(const std::string & path);
    bool load_merges_txt(const std::string & path);
    bool load_added_tokens_json(const std::string & path);

    std::vector<int32_t> tokenize(const std::string & text) const;

    static std::string punc_norm(const std::string & text);
};
