#!/usr/bin/env python3
"""Compare C++ GPT-2 BPE tokenizer output against Python reference."""

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer
from chatterbox.tts_turbo import punc_norm


TEST_CASES = [
    "Hello from ggml.",
    "The quick brown fox jumps over the lazy dog.",
    "Oh, that is hilarious! Anyway, we do have a new model in store.",
    "Good morning! Welcome to the ggml text to speech engine. I hope you are having a wonderful day.",
    "She sells sea shells by the sea shore on a sunny summer afternoon.",
    "Let's test: numbers 1, 2, 3, and punctuation... all good?",
    "He said \u201cgoodbye\u201d\u2014then left.",
    "Oh, that's hilarious! [laugh] Um anyway, we do have a new model in store. [chuckle]",
    "Simple test.",
    "This is a more complex sentence with multiple phrases, punctuation marks, and various words.",
]


def c_tokenize(binary: Path, tokenizer_dir: Path, text: str) -> list[int]:
    """Call the chatterbox binary with --dump-tokens to get C++ tokens."""
    result = subprocess.run(
        [str(binary), "--tokenizer-dir", str(tokenizer_dir),
         "--text", text, "--dump-tokens-only"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return []
    line = result.stdout.strip()
    if not line:
        return []
    return [int(x) for x in line.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))

    passed = 0
    failed = 0
    for text in TEST_CASES:
        normalized = punc_norm(text)
        py_ids = tok(normalized, return_tensors="pt").input_ids.squeeze(0).tolist()
        cpp_ids = c_tokenize(args.binary, args.tokenizer_dir, text)

        match = py_ids == cpp_ids
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {text!r}")
        print(f"       normalized: {normalized!r}")
        print(f"       python: {py_ids}")
        print(f"       cpp:    {cpp_ids}")
        if not match:
            print(f"       diff at first mismatch:")
            for i, (p, c) in enumerate(zip(py_ids, cpp_ids)):
                if p != c:
                    print(f"         pos {i}: py={p} ({tok.decode([p])!r}) vs cpp={c} ({tok.decode([c])!r})")
                    break
            if len(py_ids) != len(cpp_ids):
                print(f"       length diff: py={len(py_ids)} cpp={len(cpp_ids)}")
        print()

    print(f"{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
