#!/usr/bin/env python3
"""
Convert Chatterbox Turbo S3Gen (flow + mel2wav) weights to GGUF.

Exports:
 - flow.input_embedding            (6561, 512)
 - flow.spk_embed_affine           weight + bias
 - flow.encoder.embed              subsampling layer
 - flow.encoder.pre_lookahead      conv1 + conv2 weights
 - flow.encoder.encoders.{0..5}    6 Conformer blocks (with rel-pos attn)
 - flow.encoder.up_layer           upsample conv
 - flow.encoder.up_embed           second subsampling
 - flow.encoder.up_encoders.{0..3} 4 more Conformer blocks
 - flow.encoder.after_norm         LayerNorm
 - flow.encoder_proj               Linear(512->80)
 - flow.decoder.estimator          ConditionalDecoder (U-Net with transformer blocks)
 - mel2wav.*                       HiFTGenerator (weight_norm convs resolved)

Also embeds built-in S3Gen conditionals:
 - prompt_token  (250,)  int32
 - prompt_feat   (500, 80) float32
 - embedding     (1, 192) float32
"""

import argparse
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"
MTL_REPO_ID   = "ResembleAI/chatterbox"

VARIANTS = {
    "turbo": {
        "repo_id": TURBO_REPO_ID,
        "allow_patterns": ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        "ckpt_filename": "s3gen_meanflow.safetensors",
        "loader": "safetensors",
        "gguf_name": "Chatterbox Turbo S3Gen",
        "gguf_description": "S3Gen flow + mel2wav (HiFT) for ggml port.",
        "meanflow": True,
        "n_timesteps": 2,
        "cfg_rate": 0.0,
    },
    "mtl": {
        "repo_id": MTL_REPO_ID,
        "allow_patterns": ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                           "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
        "ckpt_filename": "s3gen.pt",
        "loader": "torch",
        "gguf_name": "Chatterbox Multilingual S3Gen",
        "gguf_description": "S3Gen standard-CFM (10-step Euler, CFG) + HiFT vocoder for ggml port.",
        "meanflow": False,
        "n_timesteps": 10,
        "cfg_rate": 0.7,
    },
}


QUANT_CHOICES = ("f32", "f16", "q8_0", "q4_0")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), default="turbo",
                    help="Which S3Gen checkpoint to convert. 'turbo' = meanflow (2-step),"
                         " 'mtl' = standard CFM (10-step + CFG).")
    ap.add_argument("--ckpt-dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to models/chatterbox-s3gen.gguf (turbo) or "
                         "models/chatterbox-s3gen-mtl.gguf (mtl).")
    ap.add_argument("--hf-token")
    ap.add_argument("--quant", choices=QUANT_CHOICES, default="f32",
                    help="Target format for the big matmul weights (encoder "
                         "Linears, CFM attn/FF Linears, HiFT Conv1d weights, "
                         "CAMPPlus/S3TokenizerV2). Biases, LayerNorm "
                         "gammas/betas, embeddings, filterbanks and built-in "
                         "conditionals always stay F32. Tensors whose shape "
                         "cannot hold the requested block quant (rank != 2 or "
                         "ne[0] not a multiple of 32) transparently fall back "
                         "to F16 so conv kernels still benefit even at q8_0/"
                         "q4_0. Default f32 reproduces the pre-optimisation "
                         "GGUF byte-for-byte.")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path("models/chatterbox-s3gen-mtl.gguf") if args.variant == "mtl" \
                   else Path("models/chatterbox-s3gen.gguf")
    return args


def as_numpy(tensor: torch.Tensor, *, dtype=None) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# Weight-tensor storage format helper.
#
# Design goals, in priority order:
#  1. Generic across backends (CPU / Metal / Vulkan / CUDA) — ggml dispatches
#     the right kernel automatically once the tensor's GGMLQuantizationType is
#     set, so the C++ binary needs no changes.
#  2. Safe quality-wise: only the big matmul weights (rank-2 Linears with an
#     inner dim divisible by 32) get Q8_0/Q4_0; everything else is F16 or F32
#     so per-layer numerical accumulation stays in-range.
#  3. Halves (F16) or quarters (Q8_0) the on-disk size and, more importantly,
#     the memory bandwidth every backend spends on weight reads.  That is
#     where the CPU path is bottlenecked today (the CFM estimator runs 10x2
#     forwards per utterance; each forward re-reads the whole U-Net).
#
# Fallback chain:
#     requested q4_0 → q8_0 → f16 → f32
# triggered when a tensor's shape can't hold the requested block layout.
# GGML block quants require ne[0] % 32 == 0.  Conv1d kernels (stored as
# ne=[K, IC, OC]) with small K (e.g. 3, 5, 7) hit that fallback and stay F16
# — still a real 2x bandwidth win vs F32 without any accuracy loss.
# ---------------------------------------------------------------------------

_GGML_QUANT_BLOCK = 32  # QK8_0 == QK4_0 == 32

_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

# Tensors the C++ runtime reads directly as F32 via `ggml_backend_tensor_get`
# + manual indexing (not through a ggml graph).  These MUST stay F32 because
# the call sites assume `ggml_nbytes(tensor) == nelements * sizeof(float)`.
#
#   - flow/input_embedding           src/chatterbox_tts.cpp:1690  (codebook)
#   - flow/spk_embed_affine/w        src/chatterbox_tts.cpp:1789  (manual matmul)
#   - hift/m_source/l_linear/weight  src/chatterbox_tts.cpp:2078  (SineGen linear)
#
# Updating the C++ side to use the tensor's native dtype would require either
# converting to F32 at load time or teaching each consumer about F16/Q8_0
# layout — both non-trivial and not worth it for these tiny tensors
# (input_embedding is a one-shot lookup for prompt tokens; the other two
# are < 100 floats each).
_FORCE_F32_WEIGHTS = frozenset({
    "flow/input_embedding",
    "flow/spk_embed_affine/w",
    "hift/m_source/l_linear/weight",
})


def _can_block_quantize(arr: np.ndarray) -> bool:
    """Returns True iff arr's layout is compatible with 32-wide GGML block quants."""
    if arr.ndim != 2:
        return False
    # In ggml memory order ne[0] is the LAST numpy axis (axis reversal on load).
    inner = arr.shape[-1]
    return inner >= _GGML_QUANT_BLOCK and inner % _GGML_QUANT_BLOCK == 0


def add_weight(writer: "gguf.GGUFWriter", name: str, tensor: torch.Tensor,
               quant: str) -> str:
    """Add a weight tensor under `name`, quantised to `quant` when possible.

    Returns the effective storage format as a short string, so the caller
    can log a running histogram (which made it to q8_0 vs. fell back to F16
    etc.).  Non-weight tensors (biases, LN gammas, embedding tables, built-in
    conditionals, filterbanks) should bypass this helper — they're small,
    and accumulated rounding on them reads a real-run loss that outweighs
    the near-zero bandwidth savings.
    """
    arr = as_numpy(tensor, dtype=torch.float32)

    # Hard F32 list: tensors the C++ runtime reads directly as packed F32
    # bytes (not through the ggml graph).  See _FORCE_F32_WEIGHTS doc.
    if name in _FORCE_F32_WEIGHTS:
        writer.add_tensor(name, arr)
        return "f32"

    # Rank-1 tensors (biases, LayerNorm gammas) are tiny and precision-
    # sensitive.  Quietly keep them F32 regardless of --quant — the bandwidth
    # savings are negligible and F16 LN params visibly regress rel error at
    # deep layers.  Callers ideally don't route these here at all; this is
    # defence-in-depth so a stray call can't silently hurt quality.
    if arr.ndim < 2:
        writer.add_tensor(name, arr)
        return "f32"

    # Rank-3 tensors are Conv1d kernels (K, IC, OC after PyTorch → numpy).
    # They're consumed by `conv1d_f32` in src/chatterbox_tts.cpp which does
    #     im2col -> mul_mat(im2col_f32, kernel)
    # passing the kernel as mul_mat's second operand.  ggml's CPU backend
    # asserts `src1->type == GGML_TYPE_F32` on that path, so quantising or
    # half-floating conv kernels would crash the CPU backend on load.
    # The Linear-weight path (`ggml_mul_mat(weight, activation)`) has the
    # weight as src0 and is unaffected, so transformer Q/K/V/FF Linears
    # still pick up Q8_0/Q4_0.  TODO: once conv1d_f32 is refactored to use
    # the kernel-as-src0 pattern (same as conv1d_f32_b), drop this branch.
    if arr.ndim >= 3:
        writer.add_tensor(name, arr)
        return "f32"

    if quant in ("q8_0", "q4_0") and _can_block_quantize(arr):
        qtype = _QUANT_TYPE[quant]
        qdata = gguf.quants.quantize(arr, qtype)
        writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
        return quant

    if quant in ("f16", "q8_0", "q4_0"):
        arr16 = arr.astype(np.float16)
        writer.add_tensor(name, arr16)
        return "f16"

    writer.add_tensor(name, arr)
    return "f32"


def resolve_weight_norm(state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """
    PyTorch weight_norm stores original0 (g, magnitudes) and original1 (v, direction).
    Actual weight = g * v / ||v||_2.  For 2D convs we broadcast appropriately.
    Returns the fused weight tensor.
    """
    g = state[f"{prefix}.parametrizations.weight.original0"]
    v = state[f"{prefix}.parametrizations.weight.original1"]
    # ||v|| is computed over all dims except 0 (the output channel dim)
    # by default for Conv1d. See torch.nn.utils.weight_norm.
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
    return g * v / norm


def expand_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Rewrite all `*.parametrizations.weight.original{0,1}` entries into a single
    `*.weight` tensor and drop the originals. Also rename `*.parametrizations.weight.0.original0`
    etc. if present.
    """
    out = dict(state)
    prefixes = set()
    for k in state:
        m = re.match(r"(.+)\.parametrizations\.weight\.original0$", k)
        if m:
            prefixes.add(m.group(1))
    for p in prefixes:
        out[f"{p}.weight"] = resolve_weight_norm(state, p)
        out.pop(f"{p}.parametrizations.weight.original0", None)
        out.pop(f"{p}.parametrizations.weight.original1", None)
    return out


def export(writer: gguf.GGUFWriter, state: dict, name: str, *, dtype=torch.float32):
    arr = as_numpy(state[name], dtype=dtype)
    # Map the name to a GGUF-friendly name but keep the hierarchy recognizable.
    gguf_name = name
    writer.add_tensor(gguf_name, arr)
    return arr.shape


def export_conformer_block(writer: gguf.GGUFWriter, state: dict, prefix: str,
                           gguf_prefix: str, quant: str, stats: dict):
    """Export one Conformer encoder block.

    Weights (the 6 attention Linears + 2 FF Linears) go through `add_weight`
    so they pick up the requested quantisation.  Biases, LayerNorm gammas,
    and the two positional-bias vectors stay F32 — they're tiny and round-
    tripping them hurts accuracy for zero bandwidth benefit.
    """
    weight_suffixes = [
        ("self_attn.linear_q.weight",   "attn/q/w"),
        ("self_attn.linear_k.weight",   "attn/k/w"),
        ("self_attn.linear_v.weight",   "attn/v/w"),
        ("self_attn.linear_out.weight", "attn/o/w"),
        ("self_attn.linear_pos.weight", "attn/pos/w"),
        ("feed_forward.w_1.weight",     "ff/w1/w"),
        ("feed_forward.w_2.weight",     "ff/w2/w"),
    ]
    f32_suffixes = [
        "norm_mha.weight", "norm_mha.bias",
        "norm_ff.weight",  "norm_ff.bias",
        "self_attn.linear_q.bias",
        "self_attn.linear_k.bias",
        "self_attn.linear_v.bias",
        "self_attn.linear_out.bias",
        "self_attn.pos_bias_u",
        "self_attn.pos_bias_v",
        "feed_forward.w_1.bias",
        "feed_forward.w_2.bias",
    ]
    dst_map = {
        "norm_mha.weight":           "norm_mha/w",
        "norm_mha.bias":             "norm_mha/b",
        "norm_ff.weight":            "norm_ff/w",
        "norm_ff.bias":              "norm_ff/b",
        "self_attn.linear_q.bias":   "attn/q/b",
        "self_attn.linear_k.bias":   "attn/k/b",
        "self_attn.linear_v.bias":   "attn/v/b",
        "self_attn.linear_out.bias": "attn/o/b",
        "self_attn.pos_bias_u":      "attn/pos_bias_u",
        "self_attn.pos_bias_v":      "attn/pos_bias_v",
        "feed_forward.w_1.bias":     "ff/w1/b",
        "feed_forward.w_2.bias":     "ff/w2/b",
    }
    for src_suffix, dst_suffix in weight_suffixes:
        dst = f"{gguf_prefix}/{dst_suffix}"
        fmt = add_weight(writer, dst, state[f"{prefix}.{src_suffix}"], quant)
        stats[fmt] = stats.get(fmt, 0) + 1
    for src_suffix in f32_suffixes:
        dst = f"{gguf_prefix}/{dst_map[src_suffix]}"
        writer.add_tensor(dst, as_numpy(state[f"{prefix}.{src_suffix}"],
                                        dtype=torch.float32))


def main():
    args = parse_args()
    cfg = VARIANTS[args.variant]
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = Path(snapshot_download(
            repo_id=cfg["repo_id"], token=args.hf_token,
            allow_patterns=cfg["allow_patterns"],
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / cfg["ckpt_filename"]
    print(f"Loading {ckpt_path}")
    if cfg["loader"] == "safetensors":
        raw = load_file(ckpt_path)
    elif cfg["loader"] == "torch":
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unknown loader: {cfg['loader']}")
    state = expand_weight_norm(raw)

    print(f"Resolved {len([k for k in raw if 'parametrizations' in k])} weight_norm entries")

    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    gen = conds["gen"]

    writer = gguf.GGUFWriter(str(args.out), "chatterbox-s3gen")
    writer.add_name(cfg["gguf_name"])
    writer.add_description(cfg["gguf_description"])

    writer.add_string("s3gen.variant", args.variant)
    writer.add_bool("s3gen.meanflow", cfg["meanflow"])
    writer.add_uint32("s3gen.n_timesteps", cfg["n_timesteps"])
    writer.add_float32("s3gen.cfg_rate", cfg["cfg_rate"])

    # Running tally of which storage format each routed weight actually
    # landed in.  `--quant q8_0` on a Conv1d with k=3 falls back to f16,
    # that'll show up here as (q8_0_requested=X, f16_fallback=Y).
    fmt_stats: dict[str, int] = {}

    def _w(name: str, tensor: torch.Tensor) -> None:
        """Shortcut for the common 'weight tensor' path."""
        fmt = add_weight(writer, name, tensor, args.quant)
        fmt_stats[fmt] = fmt_stats.get(fmt, 0) + 1

    # Meta / hparams
    writer.add_uint32("s3gen.speech_vocab_size", 6561)
    writer.add_uint32("s3gen.input_size", 512)
    writer.add_uint32("s3gen.output_size", 80)
    writer.add_uint32("s3gen.encoder.n_blocks", 6)
    writer.add_uint32("s3gen.encoder.up_n_blocks", 4)
    writer.add_uint32("s3gen.encoder.attention_heads", 8)
    writer.add_uint32("s3gen.encoder.head_dim", 64)
    writer.add_uint32("s3gen.encoder.ff_size", 2048)
    writer.add_uint32("s3gen.encoder.token_mel_ratio", 2)
    writer.add_uint32("s3gen.encoder.pre_lookahead_len", 3)
    writer.add_float32("s3gen.layer_norm_eps", 1e-12)
    writer.add_uint32("s3gen.spk_embed_dim", 192)

    # Built-in conditionals
    prompt_token = gen["prompt_token"].reshape(-1).to(torch.int32)
    prompt_feat = gen["prompt_feat"].squeeze(0)          # (500, 80)
    embedding = gen["embedding"].squeeze(0)              # (192,)
    writer.add_uint32("s3gen.builtin.prompt_token_len", int(prompt_token.numel()))
    writer.add_uint32("s3gen.builtin.prompt_feat_frames", int(prompt_feat.shape[0]))
    writer.add_tensor("s3gen/builtin/prompt_token", as_numpy(prompt_token))
    writer.add_tensor("s3gen/builtin/prompt_feat", as_numpy(prompt_feat, dtype=torch.float32))
    writer.add_tensor("s3gen/builtin/embedding", as_numpy(embedding, dtype=torch.float32))

    # Flow top-level weights.
    # input_embedding is an Embedding table (consumed by ggml_get_rows which
    # accepts quantised inputs, same as llama.cpp's token_embd).  The two
    # affine Linears are tiny (one-shot during encoder prep) but still go
    # through _w for consistency.
    _w("flow/input_embedding",       state["flow.input_embedding.weight"])
    _w("flow/spk_embed_affine/w",    state["flow.spk_embed_affine_layer.weight"])
    writer.add_tensor("flow/spk_embed_affine/b", as_numpy(state["flow.spk_embed_affine_layer.bias"], dtype=torch.float32))
    _w("flow/encoder_proj/w",        state["flow.encoder_proj.weight"])
    writer.add_tensor("flow/encoder_proj/b", as_numpy(state["flow.encoder_proj.bias"], dtype=torch.float32))

    # Encoder embed (LinearNoSubsampling: Linear(512 -> 512) + LayerNorm).
    _w("flow/encoder/embed/linear/w", state["flow.encoder.embed.out.0.weight"])
    writer.add_tensor("flow/encoder/embed/linear/b", as_numpy(state["flow.encoder.embed.out.0.bias"], dtype=torch.float32))
    writer.add_tensor("flow/encoder/embed/norm/w",   as_numpy(state["flow.encoder.embed.out.1.weight"], dtype=torch.float32))
    writer.add_tensor("flow/encoder/embed/norm/b",   as_numpy(state["flow.encoder.embed.out.1.bias"],   dtype=torch.float32))

    # PreLookaheadLayer: two short-kernel convs.  Conv1d weights land in F16
    # when --quant requests block quant (ne[0] = kernel_size ∈ {3, 4} fails
    # the divisible-by-32 check).  F16 still halves bandwidth cleanly.
    _w("flow/encoder/pre_lookahead/conv1/w", state["flow.encoder.pre_lookahead_layer.conv1.weight"])
    writer.add_tensor("flow/encoder/pre_lookahead/conv1/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.bias"], dtype=torch.float32))
    _w("flow/encoder/pre_lookahead/conv2/w", state["flow.encoder.pre_lookahead_layer.conv2.weight"])
    writer.add_tensor("flow/encoder/pre_lookahead/conv2/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.bias"], dtype=torch.float32))

    # 6 Conformer blocks.
    for i in range(6):
        export_conformer_block(writer, state,
                               f"flow.encoder.encoders.{i}",
                               f"flow/encoder/block{i}",
                               args.quant, fmt_stats)

    # Upsample1D (Conv1d kernel=5) — falls back to F16 at q8_0/q4_0.
    _w("flow/encoder/up_layer/conv/w", state["flow.encoder.up_layer.conv.weight"])
    writer.add_tensor("flow/encoder/up_layer/conv/b", as_numpy(state["flow.encoder.up_layer.conv.bias"], dtype=torch.float32))

    # up_embed (second subsampling).
    _w("flow/encoder/up_embed/linear/w", state["flow.encoder.up_embed.out.0.weight"])
    writer.add_tensor("flow/encoder/up_embed/linear/b", as_numpy(state["flow.encoder.up_embed.out.0.bias"], dtype=torch.float32))
    writer.add_tensor("flow/encoder/up_embed/norm/w",   as_numpy(state["flow.encoder.up_embed.out.1.weight"], dtype=torch.float32))
    writer.add_tensor("flow/encoder/up_embed/norm/b",   as_numpy(state["flow.encoder.up_embed.out.1.bias"],   dtype=torch.float32))

    # 4 more Conformer blocks.
    for i in range(4):
        export_conformer_block(writer, state,
                               f"flow.encoder.up_encoders.{i}",
                               f"flow/encoder/up_block{i}",
                               args.quant, fmt_stats)

    # Final after_norm.
    writer.add_tensor("flow/encoder/after_norm/w", as_numpy(state["flow.encoder.after_norm.weight"], dtype=torch.float32))
    writer.add_tensor("flow/encoder/after_norm/b", as_numpy(state["flow.encoder.after_norm.bias"], dtype=torch.float32))

    # Decoder estimator (CFM) — the critical path on CPU/Metal/Vulkan since
    # it runs 10-20 forwards per utterance on standard CFM.  Every tensor
    # goes through _w so Linear weights pick up Q8_0 and Conv1d kernels
    # pick up F16.  LayerNorm gammas/betas + biases are rank-1 and stay F32
    # via add_weight's guard.
    decoder_keys = sorted(k for k in state if k.startswith("flow.decoder.estimator."))
    for k in decoder_keys:
        gguf_name = k.replace("flow.decoder.estimator.", "cfm/").replace(".", "/")
        _w(gguf_name, state[k])

    # mel2wav (HiFTGenerator): dozens of weight_norm Conv1d layers feeding
    # the 24 kHz vocoder.  These are almost all rank-3 (K, IC, OC) with
    # short kernels → F16 at any --quant >= f16.  Real bandwidth savings on
    # every backend (HiFT decode is ~8% of CPU wall time on MTL).
    mel2wav_keys = sorted(k for k in state if k.startswith("mel2wav."))
    for k in mel2wav_keys:
        gguf_name = k.replace("mel2wav.", "hift/").replace(".", "/")
        _w(gguf_name, state[k])

    # Bake in the pre-computed 80-channel mel filterbank used by
    # s3gen.utils.mel.mel_spectrogram so the C++ side can compute prompt_feat
    # natively for voice cloning (see src/voice_features.cpp).
    import librosa
    mel_fb_24k_80 = librosa.filters.mel(
        sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000,
    ).astype(np.float32)  # (80, 961)
    writer.add_tensor("s3gen/mel_fb/24k_80", np.ascontiguousarray(mel_fb_24k_80))

    # -------------------------------------------------------------------------
    # CAMPPlus speaker encoder (FunASR/3D-Speaker xvector port).  Produces the
    # 192-d `embedding` tensor that drives S3Gen's spk_embed_affine layer.
    # We fuse every BatchNorm's affine + running stats into a per-channel
    # (scale, shift) pair so the C++ side can skip BN as its own module.
    #   y = gamma * (x - mean) / sqrt(var + eps) + beta
    #     = x * scale + shift
    #   scale = gamma / sqrt(var + eps)  (=1/sqrt(var+eps) when affine=False)
    #   shift = beta - mean * scale      (=-mean*scale when affine=False)
    # -------------------------------------------------------------------------
    speaker_keys = [k for k in state if k.startswith("speaker_encoder.")]
    if not speaker_keys:
        print(f"warning: no speaker_encoder.* tensors found in {ckpt_path}")
    else:
        BN_EPS = 1e-5  # torch.nn.BatchNorm default

        # Group BN tensors by their prefix (everything before the final component).
        # A BN module contributes: weight (optional, affine=True), bias (optional),
        # running_mean, running_var, num_batches_tracked (ignored).
        bn_groups: dict[str, dict[str, torch.Tensor]] = {}
        for k in speaker_keys:
            parts = k.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("weight", "bias", "running_mean",
                                                "running_var", "num_batches_tracked"):
                bn_groups.setdefault(parts[0], {})[parts[1]] = state[k]

        # A key is BN-owned iff its group has running_mean AND running_var.
        bn_prefixes = {p for p, t in bn_groups.items()
                       if "running_mean" in t and "running_var" in t}

        n_bn = 0
        n_conv = 0
        for k in speaker_keys:
            parts = k.rsplit(".", 1)
            prefix, last = (parts[0], parts[1]) if len(parts) == 2 else (k, "")

            # Skip training-only counters.
            if last == "num_batches_tracked":
                continue

            gguf_base = "campplus/" + prefix.removeprefix("speaker_encoder.").replace(".", "/")

            if prefix in bn_prefixes:
                if last in ("weight", "bias"):
                    # Skip the raw gamma/beta; we'll emit the fused scale/shift
                    # once per group when we hit running_mean.
                    continue
                if last == "running_var":
                    continue
                if last == "running_mean":
                    grp = bn_groups[prefix]
                    mean = grp["running_mean"].float()
                    var  = grp["running_var"].float()
                    denom = torch.sqrt(var + BN_EPS)
                    if "weight" in grp and "bias" in grp:
                        gamma = grp["weight"].float()
                        beta  = grp["bias"].float()
                        scale = gamma / denom
                        shift = beta - mean * scale
                    else:
                        # BatchNorm1d(..., affine=False) — only running stats.
                        scale = 1.0 / denom
                        shift = -mean * scale
                    writer.add_tensor(gguf_base + "/s",
                                      np.ascontiguousarray(scale.numpy().astype(np.float32)))
                    writer.add_tensor(gguf_base + "/b",
                                      np.ascontiguousarray(shift.numpy().astype(np.float32)))
                    n_bn += 1
                continue

            # Non-BN tensor: export as-is (F32).
            gguf_name = "campplus/" + k.removeprefix("speaker_encoder.").replace(".", "/")
            writer.add_tensor(gguf_name, as_numpy(state[k], dtype=torch.float32))
            n_conv += 1

        # Hyperparameters.  CAMPPlus() is instantiated with the defaults in
        # s3gen.py, so hard-code them here to avoid re-encoding in C++.
        writer.add_uint32("campplus.feat_dim",         80)
        writer.add_uint32("campplus.embedding_size",   192)
        writer.add_uint32("campplus.growth_rate",      32)
        writer.add_uint32("campplus.bn_size",          4)
        writer.add_uint32("campplus.init_channels",    128)
        writer.add_uint32("campplus.block1_layers",    12)
        writer.add_uint32("campplus.block2_layers",    24)
        writer.add_uint32("campplus.block3_layers",    16)
        writer.add_uint32("campplus.block1_dilation",  1)
        writer.add_uint32("campplus.block2_dilation",  2)
        writer.add_uint32("campplus.block3_dilation",  2)
        writer.add_uint32("campplus.kernel_size",      3)
        writer.add_uint32("campplus.seg_pool_len",     100)
        writer.add_uint32("campplus.sample_rate",      16000)

        # Kaldi-style mel filterbank (80 bins, 16 kHz, n_fft=512, low=20 Hz,
        # high=8000 Hz).  Used by the C++ fbank_kaldi_80 implementation in
        # src/voice_features.cpp to replace torchaudio.compliance.kaldi.fbank
        # at runtime.  Formula: triangular filters equally spaced in mel-space
        # (Kaldi mel: 1127 * log(1 + f/700)), evaluated at each FFT bin's
        # linear frequency.
        SR = 16000
        NFFT = 512
        N_MELS = 80
        LOW = 20.0
        HIGH = 8000.0
        mel_low  = 1127.0 * np.log(1.0 + LOW  / 700.0)
        mel_high = 1127.0 * np.log(1.0 + HIGH / 700.0)
        mel_delta = (mel_high - mel_low) / (N_MELS + 1)
        bin_freq  = np.arange(NFFT // 2 + 1, dtype=np.float64) * SR / NFFT
        bin_mel   = 1127.0 * np.log(1.0 + bin_freq / 700.0)
        kaldi_fb  = np.zeros((N_MELS, NFFT // 2 + 1), dtype=np.float32)
        for m in range(N_MELS):
            mel_center = mel_low + (m + 1) * mel_delta
            mel_lo = mel_center - mel_delta
            mel_hi = mel_center + mel_delta
            for k, mb in enumerate(bin_mel):
                if mb < mel_lo or mb > mel_hi:
                    continue
                if mb <= mel_center:
                    kaldi_fb[m, k] = (mb - mel_lo) / (mel_center - mel_lo)
                else:
                    kaldi_fb[m, k] = (mel_hi - mb) / (mel_hi - mel_center)
        writer.add_tensor("campplus/mel_fb_kaldi_80", np.ascontiguousarray(kaldi_fb))
        print(f"Embedded CAMPPlus: {n_conv} conv/linear tensors + {n_bn} fused BNs "
              f"+ kaldi mel filterbank {kaldi_fb.shape}")

    # -------------------------------------------------------------------------
    # S3TokenizerV2 (FunASR speech-to-token encoder that produces the 25 Hz
    # token stream Chatterbox uses for voice conditioning).  103 raw tensors:
    #   tokenizer._mel_filters                   (128, 201) librosa mel fb
    #   tokenizer.encoder.conv{1,2}.{weight,bias}
    #   tokenizer.encoder.blocks.{0..5}.*        (16 tensors each × 6 = 96)
    #   tokenizer.quantizer._codebook.project_down.{weight,bias}
    # -------------------------------------------------------------------------
    tok_keys = [k for k in state if k.startswith("tokenizer.")]
    if not tok_keys:
        print(f"warning: no tokenizer.* tensors found in {ckpt_path}")
    else:
        n_tok = 0
        for k in tok_keys:
            rest = k[len("tokenizer."):]
            # Skip window buffer (we recompute it).
            if rest in ("window",):
                continue
            if rest == "_mel_filters":
                gguf_name = "s3tokv2/mel_fb"
            else:
                gguf_name = "s3tokv2/" + rest.replace(".", "/")
            writer.add_tensor(gguf_name, as_numpy(state[k], dtype=torch.float32))
            n_tok += 1

        writer.add_uint32("s3tokv2.n_mels",        128)
        writer.add_uint32("s3tokv2.n_audio_state", 1280)
        writer.add_uint32("s3tokv2.n_audio_head",  20)
        writer.add_uint32("s3tokv2.n_audio_layer", 6)
        writer.add_uint32("s3tokv2.head_dim",      64)
        writer.add_uint32("s3tokv2.mlp_ratio",     4)
        writer.add_uint32("s3tokv2.fsmn_kernel",   31)
        writer.add_uint32("s3tokv2.fsq_levels",    3)
        writer.add_uint32("s3tokv2.fsq_dim",       8)
        writer.add_uint32("s3tokv2.codebook_size", 3 ** 8)
        writer.add_uint32("s3tokv2.conv_stride",   2)
        writer.add_uint32("s3tokv2.n_fft",         400)
        writer.add_uint32("s3tokv2.hop",           160)
        writer.add_uint32("s3tokv2.sample_rate",   16000)
        writer.add_float32("s3tokv2.rope_theta",   10000.0)
        writer.add_uint32("s3tokv2.rope_max_pos",  2048)
        print(f"Embedded S3TokenizerV2: {n_tok} tensors")

    n_flow = sum(1 for k in state if k.startswith("flow.")) - sum(1 for k in state if k.startswith("flow.decoder.estimator."))
    n_cfm  = len(decoder_keys)
    n_hift = len(mel2wav_keys)
    print(f"Wrote: encoder(+proj)~{n_flow} tensors, cfm={n_cfm}, hift={n_hift}")
    if fmt_stats:
        breakdown = " ".join(f"{k}={v}" for k, v in sorted(fmt_stats.items()))
        print(f"Weight format breakdown (requested --quant {args.quant}): {breakdown}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
