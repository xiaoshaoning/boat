#!/usr/bin/env python3
"""Check that all git-tracked text files are UTF-8 without BOM and free of
common mojibake artifacts (GBK/GB2312 reads or double-encoded content).

The repository policy (see CLAUDE.md / .editorconfig) requires UTF-8 without
BOM for every text file. Files that are opened or saved as GBK/GB2312 on
Windows machines produce exactly the signatures this script detects.

Exit code is 0 when the tree is clean, 1 otherwise.

Usage:
    python3 scripts/check_encoding.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Binary files that are legitimately non-UTF-8 (weights, archives, images).
BINARY_EXTS = {
    ".pt", ".pth", ".gguf", ".safetensors", ".onnx", ".bin", ".f32", ".f16",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".whl",
    ".o", ".a", ".so", ".dll", ".exe", ".obj", ".lib",
}

# Signatures of UTF-8 bytes that were decoded as GBK and re-saved as UTF-8.
MOJIBAKE_SEQUENCES = ("\u951f\u65a4\u62f7",)
MOJIBAKE_CHARS = "\u9480\u9225\u922d\u9224\u9227\u942e\u92f1\ufffd"


def git_tracked_files():
    out = subprocess.run(
        ["git", "-C", ROOT, "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    return [f for f in out.decode("utf-8").split("\0") if f]


def check_file(relpath):
    path = os.path.join(ROOT, relpath)
    with open(path, "rb") as fh:
        data = fh.read()

    if os.path.splitext(relpath)[1].lower() in BINARY_EXTS:
        return []

    problems = []
    if data.startswith(b"\xef\xbb\xbf"):
        problems.append("UTF-8 BOM")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append("not valid UTF-8 (%s)" % exc)
        return problems

    for seq in MOJIBAKE_SEQUENCES:
        if seq in text:
            problems.append("mojibake sequence %r" % seq)
    for ch in MOJIBAKE_CHARS:
        if ch in text:
            problems.append("mojibake char U+%04X" % ord(ch))
    return problems


def main():
    files = git_tracked_files()
    bad = []
    for relpath in files:
        for problem in check_file(relpath):
            bad.append((relpath, problem))
    for relpath, problem in bad:
        print("%s: %s" % (relpath, problem), file=sys.stderr)
    if bad:
        print("encoding check FAILED: %d file(s)" % len(bad), file=sys.stderr)
        return 1
    print("encoding check OK: %d file(s)" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())