# qvac-chatterbox.cpp

Chatterbox Turbo speech synthesis (Resemble AI, MIT) ported to
[`ggml`](https://github.com/ggml-org/ggml). See [`PROGRESS.md`](PROGRESS.md)
for the full development journal.

## Layout

```
qvac-chatterbox.cpp/
  ggml/                    ← pristine ggml clone (not tracked in git)
  src/
    main.cpp               T3 ggml runner
    gpt2_bpe.{h,cpp}       self-contained GPT-2 BPE tokenizer
    test_s3gen.cpp         staged S3Gen verification harness
    npy.h                  minimal .npy loader
  scripts/
    convert-t3-turbo-to-gguf.py
    convert-s3gen-to-gguf.py
    dump-s3gen-reference.py
    reference-t3-turbo.py
    compare-tokenizer.py
  models/                  GGUF weights (generated, not tracked)
  CMakeLists.txt
```

## Setup

Clone `ggml` alongside this repo:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Convert

```bash
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
```

## Run

```bash
./build/chatterbox \
  --model models/chatterbox-t3-turbo.gguf \
  --tokens-file text_tokens.txt \
  --output speech_tokens.txt
```

## Compare with reference

```bash
python scripts/reference-t3-turbo.py \
  --text "Hello from ggml." \
  --out-dir artifacts \
  --cpp-bin ./build/chatterbox \
  --cpp-model models/chatterbox-t3-turbo.gguf
```
