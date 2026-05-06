# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

| Patch | When you need it |
|--------|------------------|
| `ggml-backend-reg-filename-prefix.patch` | Renaming the bundled ggml libraries (the default `TTS_CPP_GGML_LIB_PREFIX=ON` emits `libspeech-ggml-*`). Teaches `ggml_backend_load_best()` to honour `GGML_BACKEND_DL_PROJECT_PREFIX` so the runtime backend-discovery walk finds the renamed `.so` / `.dll` files. **No-op when the macro is undefined**, so this is also safe to ship for `TTS_CPP_GGML_LIB_PREFIX=OFF` builds. |
| `ggml-metal-chatterbox-ops.patch` | Building with **Metal** (Apple Silicon T3 + full pipeline). |
| `ggml-opencl-allow-non-adreno.patch` | Bringing up `-DGGML_OPENCL=ON` builds outside an Adreno-only environment (NVIDIA / AMD / Apple desktop dev / CI parity testing). Gated behind `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1`; **no-op on real Adreno targets**. |
| `ggml-opencl-chatterbox-ops.patch` | Building with **OpenCL** — `CONV_TRANSPOSE_1D` for HiFT, `SIN`, `COS`, `LEAKY_RELU`, `ABS`, `ELU`. |
| `ggml-opencl-program-binary-cache.patch` | Building with **OpenCL** — persistent on-disk OpenCL kernel-binary cache. Removes the multi-second `clBuildProgram` wave on every cold start (Adreno / Mesa / Mali / iGPU). Honours `$GGML_OPENCL_CACHE_DIR` with `$XDG_CACHE_HOME/ggml/opencl` → `$HOME/.cache/ggml/opencl` fallbacks. Opt-out via `GGML_OPENCL_CACHE_DIR=""`. |
| `ggml-vulkan-pipeline-cache.patch` | Building with **Vulkan** — opt-in persistent `VkPipelineCache` keyed by `<vendorID>-<deviceID>-<driverVersion>`. Recovers ~91 % of the cold→warm gap on the first warm run. Disabled by `GGML_VK_PIPELINE_CACHE_DIR=""`. |
| `ggml-vulkan-eager-cache-save.patch` | Building with **Vulkan** — write back the pipeline cache after every `ggml_vk_load_shaders` compile batch (crash-safety against SIGKILL/abort losing freshly compiled pipelines). Stacks on the previous patch. |
| (none) | **CPU** / **CUDA** only — the filename-prefix patch above is a strict no-op when `TTS_CPP_GGML_LIB_PREFIX=OFF`, and the OpenCL / Vulkan / Metal patches are inert when the corresponding backend is disabled. |

`scripts/setup-ggml.sh` discovers every `patches/ggml-*.patch` via a
lex-sorted glob and applies them in order.  Each is inert when you
configure without the corresponding backend
(`GGML_METAL=ON` / `GGML_OPENCL=ON` / `GGML_VULKAN=ON`) or
without `TTS_CPP_GGML_LIB_PREFIX=ON`.

## Apply

The top-level [`scripts/setup-ggml.sh`](../scripts/setup-ggml.sh) does
everything for you:

```bash
# From the repo root.  Clones ggml if needed, checks out the pinned
# commit, and applies every patch under patches/ in lexicographic
# order.  Idempotent: re-running is a no-op once the patches are
# already applied.
./scripts/setup-ggml.sh
```

Then configure + build as usual, for example:

```bash
# Metal (macOS)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON

# Vulkan (any platform)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON

# OpenCL: Adreno (Android) target
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON

# OpenCL: NVIDIA / AMD / Apple desktop (dev / CI parity testing) —
# Adreno-tuned matmul kernels OFF, generic OpenCL paths only:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
```

If you'd rather run the steps by hand (e.g. to pin a different upstream
commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git checkout $GGML_COMMIT
for p in ../patches/ggml-*.patch; do git apply "$p"; done
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth — bump it when re-generating patches against
a newer upstream ggml.  To confirm everything applied cleanly:

```bash
(cd ggml && git status --short)
# Expected: a handful of modified files, depending on which backends
# the patches touch.
```

Skip `setup-ggml.sh` only if you use `-DTTS_CPP_USE_SYSTEM_GGML=ON`
with a pre-patched system ggml (the qvac speech-stack
`qvac-ext-ggml/speech` port ships these patches pre-applied);
otherwise the pin + patches keep builds deterministic.

## `ggml-backend-reg-filename-prefix.patch`

Base commit: `58c38058` (`sync : llama.cpp`, 2026-04-09).

Adds a single compile-time switch `GGML_BACKEND_DL_PROJECT_PREFIX` to
`ggml_backend_load_best()` so the runtime backend-discovery walk can
be retargeted at the filename prefix used by a host project that
renames the bundled `libggml-*` files to avoid colliding with another
consumer's `libggml-*` files in the same host process.

`tts-cpp` ships its bundled ggml backends as `libspeech-ggml-*.{so,dll}`
(CMake option `TTS_CPP_GGML_LIB_PREFIX=ON`, default), shared with
parakeet.cpp so the QVAC speech stack co-vendors a single ggml file
set.  Without this patch, the rename works at link time but
`ggml_backend_load_best()` still searches for `libggml-*.so` /
`ggml-*.dll`, so under `GGML_BACKEND_DL=ON` (the default on Android)
the renamed files are on disk but never discovered and Vulkan / OpenCL /
CUDA backends silently fail to load.

When `GGML_BACKEND_DL_PROJECT_PREFIX` is undefined the patch is a
strict no-op, so `TTS_CPP_GGML_LIB_PREFIX=OFF` builds (and all
single-consumer hosts) get behaviour byte-equal to upstream.

## `ggml-metal-chatterbox-ops.patch`

Base commit: `58c38058`.

Fixes three gaps in ggml-metal that make Chatterbox unusable or very slow
on Metal:

| Symptom                                       | Root cause in ggml-metal                          | What this patch does                                           |
|-----------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|
| T3 crashes: `unsupported op 'DIAG_MASK_INF'`  | No op entry / no kernel                            | Adds `kernel_diag_mask_inf_f32`, dispatcher, `supports_op` case|
| S3Gen crashes: `unsupported op 'PAD'` when any front-pad (`lp0..lp3`) is non-zero | Kernel only supports tail padding; `supports_op` rejects non-zero front pads | Extends `kernel_pad_f32` + `ggml_metal_kargs_pad` to honour `lp0..lp3` and drops the rejection  |
| HiFT decode is ~100× slower than CPU          | `kernel_conv_transpose_1d` is scalar: 1 thread per output pixel iterating over *all* `IC * IL` inputs, with most of the work inside a conditional | Tighten the input-position range to the few that contribute (`i_min..i_max`) and parallelise `IC` across a 32-thread simdgroup with `simd_sum` reduction |
| T3 step does `mul_mv + bin_fuse(add)` / `mul_mv + bin_fuse(add+add)` per linear layer | `mul_mv` and the following bias / bias+residual adds are separate Metal kernels even though Vulkan fuses the same patterns (`ggml_vk_can_fuse` + `Fuse0` / `Fuse1` shader bindings) | Fuse `mul_mat + add(bias)` and `mul_mat + add(bias) + add(residual)` for the Q-variant mat-vec kernels (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0) via two function constants (`FC_mul_mv_has_bias`, `FC_mul_mv_has_residual`) and a `helper_mv_add_bias<NR0>` post-pass.  The op encoder tries `{MUL_MAT, ADD, ADD}` first and falls back to `{MUL_MAT, ADD}`; `n_fuse` tells the dispatcher how many nodes to consume |

Measured on M3 Ultra, `hift_decode` at HiFT-realistic shapes:
- Before: ~15 000 ms
- After:    ~350 ms (≈ 40× speedup; end-to-end `gen_RTF` goes from
  unusable → 0.19 on F16)

Correctness is validated against the ggml CPU backend by the
`test-metal-ops` binary built in the parent repo (Metal builds only).
Run it after rebuilding:

```bash
./build/test-metal-ops
# Expected: "diag_mask_inf / pad_ext / conv_transpose_1d: PASS"
```

## `ggml-opencl-allow-non-adreno.patch`

Base commit: `58c38058`.

Fixes two gaps in `ggml-opencl` that make `-DGGML_OPENCL=ON` builds
impossible to bring up outside an Adreno-only environment:

- `ggml_cl2_init()` whitelists `Adreno` / `Qualcomm` / `Intel` and
  returns `nullptr` for everything else.  Even with
  `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`, a non-Adreno GPU never reaches
  the generic kernels.  Set `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1` to opt
  the device through with `GPU_FAMILY::UNKNOWN`; we additionally
  require `cl_intel_required_subgroup_size` *or*
  `cl_qcom_reqd_sub_group_size`, so AMD / NVIDIA still fall back to
  host instead of crashing in `clBuildProgram`.

- `ggml_backend_opencl_init()` calls `ggml_backend_reg_dev_get(reg, 0)`
  unconditionally; when device discovery cleared the list, that
  asserts.  Patch checks `dev_count == 0` first and returns `nullptr`
  so the host-side fallback path actually runs.

The patch is **strictly additive** for real Adreno targets:
`gpu_family == ADRENO` is computed exactly as before, the Adreno
shuffle / large-buffer paths still trigger when (and only when) the
device is Adreno, and without `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1` the
non-Adreno reject path is byte-equal to upstream.

The intended audience is dev / CI parity testing; it is **not**
intended to ship a fast OpenCL path on NVIDIA / AMD / Apple desktops
(CUDA / Vulkan / Metal are far better suited there).

## `ggml-opencl-chatterbox-ops.patch`

Base commit: `58c38058` (same pin as the Metal patch).

Extends `ggml-opencl` so Chatterbox's S3Gen + **HiFT** path can run on
OpenCL (e.g. Qualcomm Adreno) instead of failing on missing ops:

| What | Purpose |
|------|---------|
| `CONV_TRANSPOSE_1D` | f32 and f16-kernel + f32 input kernels, dispatch + `supports_op` |
| `GGML_OP_SIN`, `GGML_OP_COS` | Element-wise trig kernels, dispatch + `supports_op` |
| `GGML_OP_LEAKY_RELU` | `kernel_leaky_relu` in `relu.cl`, dispatch + `supports_op` |
| `GGML_UNARY_OP_ABS`, `GGML_UNARY_OP_ELU` | f32 unary kernels for the HiFT f0 predictor path |
| `ggml-opencl.h` | Document that `ggml_backend_opencl_init` may return NULL when no device |
| Build | Register new `.cl` sources in `CMakeLists.txt` for embed |

Regenerate from a throwaway `ggml` worktree at `GGML_COMMIT` after editing
upstream:

```bash
# From cherry-picked commits or a branch:
(cd ggml && git diff 58c38058..your-branch) > patches/ggml-opencl-chatterbox-ops.patch
```

## `ggml-opencl-program-binary-cache.patch`

Base commit: `58c38058`.

Adds a persistent on-disk cache for compiled OpenCL kernel binaries.
Upstream `build_program_from_source()` calls `clCreateProgramWithSource`
+ `clBuildProgram` on every cold start, re-paying the driver's
shader-compile wave (multiple seconds on Adreno / Mesa / Mali; tens of
ms on most desktop drivers).  This patch tries
`clCreateProgramWithBinary` against a device-specific cache blob
whenever one exists, and persists every freshly-compiled program back
to disk on miss.

- **Cache key**: `<src_hash>_<opts_hash>_<driver_hash>_<dev_name_hash>_<dev_ver_hash>.bin`
  (FNV-1a-64 per component).  Driver upgrades or moving to a different
  device silently invalidate the cache because either `driver_hash` or
  `dev_*_hash` changes.
- **Atomic writes**: dump to `<path>.tmp` then `rename(2)`; concurrent
  processes can't read a half-written file.
- **Stale-cache handling**: `CL_INVALID_BINARY` (or a subsequent
  `clBuildProgram` failure) falls through to source compile and
  overwrites the bad blob on the next run.
- **Opt-out**: `GGML_OPENCL_CACHE_DIR=""` (literal empty string)
  short-circuits both read and write paths.  Useful for benchmarking
  cold-start cost.

Cache directory resolution: `$GGML_OPENCL_CACHE_DIR` →
`$XDG_CACHE_HOME/ggml/opencl` → `$HOME/.cache/ggml/opencl`.  Each
kernel binary lands at ~10-200 KB on Adreno; 88 kernels × ~50 KB
average ≈ 4-5 MB on disk per device per process family.

## `ggml-vulkan-pipeline-cache.patch` + `ggml-vulkan-eager-cache-save.patch`

Both patches at base commit `58c38058`.

Opt-in persistent `VkPipelineCache` across processes for `-DGGML_VULKAN=ON`
builds, plus a write-back pass after every `ggml_vk_load_shaders` batch
so a SIGKILL/abort doesn't lose freshly-compiled pipelines.  Cache file
is keyed on `<vendorID>-<deviceID>-<driverVersion>`, and Vulkan validates
the blob header so stale cache files (driver upgrade, shader-bundle
change) are silently ignored.

Enabled by setting `GGML_VK_PIPELINE_CACHE_DIR` to a non-empty path;
when unset or empty the patches are no-op.  Recovers ~91 % of the
cold→warm shader-compile gap on the first warm run on drivers without
an aggressive per-app system cache (Mesa / RADV, Android Adreno / Mali,
fresh NVIDIA installs, containers).

## Dropping the patches

If upstream ggml merges equivalent fixes, delete the patch file(s);
`scripts/setup-ggml.sh`'s glob discovers whatever's in `patches/`
without further edits.  The C++ side of Chatterbox uses only ops that
ggml already supports natively, so nothing else needs to change.
