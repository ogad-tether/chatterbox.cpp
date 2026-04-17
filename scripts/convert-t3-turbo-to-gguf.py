#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


REPO_ID = "ResembleAI/chatterbox-turbo"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]

TEXT_VOCAB_SIZE = 50276
SPEECH_VOCAB_SIZE = 6563
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SPEAKER_EMBED_SIZE = 256
N_CTX = 8196
N_EMBD = 1024
N_HEAD = 16
N_LAYER = 24
LAYER_NORM_EPS = 1e-5

LAYER_RE = re.compile(r"^tfmr\.h\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Chatterbox Turbo T3 weights to GGUF.")
    parser.add_argument("--ckpt-dir", type=Path, help="Local checkpoint dir (downloads from HF if omitted).")
    parser.add_argument("--out", type=Path, default=Path("models/chatterbox-t3-turbo.gguf"), help="Output GGUF path.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    return parser.parse_args()


def as_numpy(tensor: torch.Tensor, *, dtype=None, transpose: bool = False) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    array = tensor.detach().cpu().numpy()
    if transpose:
        array = array.T
    return np.ascontiguousarray(array)


def map_tensor_name(name: str):
    if name == "tfmr.wte.weight":
        return None
    if name == "tfmr.wpe.weight":
        return "model/wpe", torch.float32, False
    if name == "tfmr.ln_f.weight":
        return "model/ln_f/g", torch.float32, False
    if name == "tfmr.ln_f.bias":
        return "model/ln_f/b", torch.float32, False
    if name == "text_emb.weight":
        return "chatterbox/text_emb", torch.float16, False
    if name == "speech_emb.weight":
        return "chatterbox/speech_emb", torch.float16, False
    if name == "speech_head.weight":
        return "chatterbox/speech_head", torch.float16, False
    if name == "speech_head.bias":
        return "chatterbox/speech_head_bias", torch.float32, False
    if name == "cond_enc.spkr_enc.weight":
        return "chatterbox/cond_spkr/w", torch.float32, False
    if name == "cond_enc.spkr_enc.bias":
        return "chatterbox/cond_spkr/b", torch.float32, False

    match = LAYER_RE.match(name)
    if not match:
        return None

    layer_idx = int(match.group(1))
    suffix = match.group(2)

    # GPT-2 Conv1D weights need transposing; biases and LayerNorm do not
    table = {
        "ln_1.weight": ("model/h{}/ln_1/g", torch.float32, False),
        "ln_1.bias": ("model/h{}/ln_1/b", torch.float32, False),
        "ln_2.weight": ("model/h{}/ln_2/g", torch.float32, False),
        "ln_2.bias": ("model/h{}/ln_2/b", torch.float32, False),
        "attn.c_attn.weight": ("model/h{}/attn/c_attn/w", torch.float16, True),
        "attn.c_attn.bias": ("model/h{}/attn/c_attn/b", torch.float32, False),
        "attn.c_proj.weight": ("model/h{}/attn/c_proj/w", torch.float16, True),
        "attn.c_proj.bias": ("model/h{}/attn/c_proj/b", torch.float32, False),
        "mlp.c_fc.weight": ("model/h{}/mlp/c_fc/w", torch.float16, True),
        "mlp.c_fc.bias": ("model/h{}/mlp/c_fc/b", torch.float32, False),
        "mlp.c_proj.weight": ("model/h{}/mlp/c_proj/w", torch.float16, True),
        "mlp.c_proj.bias": ("model/h{}/mlp/c_proj/b", torch.float32, False),
    }
    if suffix not in table:
        return None
    fmt, dtype, transpose = table[suffix]
    return fmt.format(layer_idx), dtype, transpose


def main() -> None:
    args = parse_args()
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = Path(snapshot_download(repo_id=REPO_ID, token=args.hf_token, allow_patterns=ALLOW_PATTERNS))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {ckpt_dir}")
    state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)

    writer = gguf.GGUFWriter(str(args.out), "chatterbox")
    writer.add_name("Chatterbox Turbo T3")
    writer.add_description("Chatterbox Turbo text-to-speech token generator for ggml.")
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_head_count(N_HEAD)
    writer.add_vocab_size(TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.n_ctx", N_CTX)
    writer.add_uint32("chatterbox.n_embd", N_EMBD)
    writer.add_uint32("chatterbox.n_head", N_HEAD)
    writer.add_uint32("chatterbox.n_layer", N_LAYER)
    writer.add_uint32("chatterbox.text_vocab_size", TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_float32("chatterbox.layer_norm_eps", LAYER_NORM_EPS)
    writer.add_string("chatterbox.variant", "t3_turbo")
    writer.add_string("chatterbox.reference_repo", REPO_ID)

    exported = 0
    ignored = []
    for name, tensor in state.items():
        mapped = map_tensor_name(name)
        if mapped is None:
            ignored.append(name)
            continue
        gguf_name, dtype, transpose = mapped
        array = as_numpy(tensor, dtype=dtype, transpose=transpose)
        writer.add_tensor(gguf_name, array)
        exported += 1
        print(f"{gguf_name:32s} {str(tuple(array.shape)):18s} {array.dtype}")

    builtin_speaker = conds["t3"]["speaker_emb"].reshape(1, SPEAKER_EMBED_SIZE)
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)

    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(builtin_speaker, dtype=torch.float32))
    writer.add_tensor("chatterbox/builtin/cond_prompt_speech_tokens", as_numpy(builtin_tokens))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {exported + 2} tensors to {args.out}")
    if ignored:
        print("\nIgnored tensors:")
        for n in ignored:
            print(f"  {n}")


if __name__ == "__main__":
    main()
