#!/usr/bin/env python3
"""Build crates/clamor-node/src/catalog.json: the models Clamor's model picker offers.

The list below is curated (name, maker, licence, a line about it, the Hugging Face repository).
Everything else is read from the real files, not typed in:
  - sizes and SHA-256 of each quantization, from Hugging Face's file listing (downloads are
    checked against these, and the catalog ships inside Clamor's signed releases);
  - the GGUF header of one file per model (only its first megabytes are downloaded):
    architecture, tokenizer, layers and widths (for speed predictions), context length,
    parameter count, and what its chat template supports (tools, thinking).
Whether Clamor can run a model is decided by clamor-node from the architecture and tokenizer
(the engine's lists), so the catalog stays true as the engine learns more.

Usage: tools/build-catalog.py   (network; a few hundred MB of headers; commit catalog.json)
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
QUANTS = ["Q8_0", "Q4_K_M"]

# id, name, maker, licence, about, repo, extra capabilities (not visible in the chat template)
MODELS = [
    ("smollm2-360m", "SmolLM2 360M Instruct", "Hugging Face", "Apache-2.0", "Tiny and very fast; good for trying Clamor out.", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", []),
    ("smollm2-1.7b", "SmolLM2 1.7B Instruct", "Hugging Face", "Apache-2.0", "Small and quick, for simple questions and short texts.", "bartowski/SmolLM2-1.7B-Instruct-GGUF", []),
    ("llama-3.2-1b", "Llama 3.2 1B Instruct", "Meta", "Llama 3.2 Community License", "Small and quick all-rounder.", "bartowski/Llama-3.2-1B-Instruct-GGUF", []),
    ("llama-3.2-3b", "Llama 3.2 3B Instruct", "Meta", "Llama 3.2 Community License", "A good everyday model for one laptop.", "bartowski/Llama-3.2-3B-Instruct-GGUF", []),
    ("llama-3.1-8b", "Llama 3.1 8B Instruct", "Meta", "Llama 3.1 Community License", "Capable all-rounder; split over two devices it is comfortable.", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", []),
    ("llama-3.3-70b", "Llama 3.3 70B Instruct", "Meta", "Llama 3.3 Community License", "Large and strong; needs a swarm with lots of memory.", "bartowski/Llama-3.3-70B-Instruct-GGUF", []),
    ("qwen2.5-0.5b", "Qwen2.5 0.5B Instruct", "Alibaba Qwen", "Apache-2.0", "Tiny and fast, many languages.", "bartowski/Qwen2.5-0.5B-Instruct-GGUF", []),
    ("qwen2.5-1.5b", "Qwen2.5 1.5B Instruct", "Alibaba Qwen", "Apache-2.0", "Small, quick, many languages.", "bartowski/Qwen2.5-1.5B-Instruct-GGUF", []),
    ("qwen2.5-3b", "Qwen2.5 3B Instruct", "Alibaba Qwen", "Qwen Research License", "Small everyday model, many languages.", "bartowski/Qwen2.5-3B-Instruct-GGUF", []),
    ("qwen2.5-7b", "Qwen2.5 7B Instruct", "Alibaba Qwen", "Apache-2.0", "Strong for its size: writing, reasoning, many languages.", "bartowski/Qwen2.5-7B-Instruct-GGUF", []),
    ("qwen2.5-14b", "Qwen2.5 14B Instruct", "Alibaba Qwen", "Apache-2.0", "Noticeably smarter; wants two or three devices.", "bartowski/Qwen2.5-14B-Instruct-GGUF", []),
    ("qwen2.5-32b", "Qwen2.5 32B Instruct", "Alibaba Qwen", "Apache-2.0", "Very capable; needs a well-stocked swarm.", "bartowski/Qwen2.5-32B-Instruct-GGUF", []),
    ("qwen2.5-coder-7b", "Qwen2.5 Coder 7B Instruct", "Alibaba Qwen", "Apache-2.0", "Writes and explains code.", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", ["code"]),
    ("qwen2.5-coder-14b", "Qwen2.5 Coder 14B Instruct", "Alibaba Qwen", "Apache-2.0", "Writes and explains code; stronger.", "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", ["code"]),
    ("qwen2.5-coder-32b", "Qwen2.5 Coder 32B Instruct", "Alibaba Qwen", "Apache-2.0", "One of the best open coding models.", "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF", ["code"]),
    ("deepseek-r1-distill-llama-8b", "DeepSeek R1 Distill Llama 8B", "DeepSeek", "MIT (Llama 3.1 base)", "Thinks step by step before answering.", "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", ["thinking"]),
    ("deepseek-r1-distill-qwen-7b", "DeepSeek R1 Distill Qwen 7B", "DeepSeek", "MIT", "Thinks step by step before answering.", "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", ["thinking"]),
    ("deepseek-r1-distill-qwen-14b", "DeepSeek R1 Distill Qwen 14B", "DeepSeek", "MIT", "Thinks step by step; stronger reasoning.", "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", ["thinking"]),
    ("qwen3-8b", "Qwen3 8B", "Alibaba Qwen", "Apache-2.0", "Tools and switchable thinking.", "Qwen/Qwen3-8B-GGUF", ["thinking"]),
    ("qwen3-14b", "Qwen3 14B", "Alibaba Qwen", "Apache-2.0", "Tools and switchable thinking; stronger.", "Qwen/Qwen3-14B-GGUF", ["thinking"]),
    ("gemma-3-4b", "Gemma 3 4B", "Google", "Gemma Terms of Use", "Small model that also understands pictures.", "bartowski/google_gemma-3-4b-it-GGUF", ["vision"]),
    ("gemma-3-12b", "Gemma 3 12B", "Google", "Gemma Terms of Use", "Understands pictures; strong all-rounder.", "bartowski/google_gemma-3-12b-it-GGUF", ["vision"]),
    ("mistral-small-24b", "Mistral Small 24B Instruct 2501", "Mistral AI", "Apache-2.0", "Fast, capable mid-size model.", "bartowski/Mistral-Small-24B-Instruct-2501-GGUF", []),
    ("nomic-embed-text-1.5", "Nomic Embed Text v1.5", "Nomic AI", "Apache-2.0", "Turns text into vectors for search (not a chat model).", "nomic-ai/nomic-embed-text-v1.5-GGUF", ["embedding"]),
]


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "clamor-catalog"}), timeout=60) as r:
        return json.load(r)


class Stream:
    """Reads a remote file from the start, only as far as needed."""

    def __init__(self, url):
        self.r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "clamor-catalog"}), timeout=60)
        self.buf = b""
        self.read_total = 0

    def take(self, n):
        while len(self.buf) < n:
            chunk = self.r.read(max(n - len(self.buf), 1 << 20))
            if not chunk:
                raise EOFError("file ended inside the header")
            self.buf += chunk
            self.read_total += len(chunk)
            if self.read_total > 256 << 20:
                raise RuntimeError("header larger than 256 MB?")
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self):
        self.r.close()


SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_header(url):
    """The key/values a GGUF file starts with, and its parameter count (from the tensor list)."""
    s = Stream(url)
    try:
        u32 = lambda: struct.unpack("<I", s.take(4))[0]
        u64 = lambda: struct.unpack("<Q", s.take(8))[0]
        string = lambda: s.take(u64()).decode("utf-8", "replace")

        def value(t, keep):
            if t in SCALARS:
                fmt = SCALARS[t]
                return struct.unpack(fmt, s.take(struct.calcsize(fmt)))[0]
            if t == 8:
                return string()
            if t == 9:
                it, n = u32(), u64()
                if it in SCALARS and not keep:
                    s.take(struct.calcsize(SCALARS[it]) * n)
                    return None
                items = [value(it, keep) for _ in range(n)]
                return items if keep else None
            raise ValueError(f"unknown GGUF type {t}")

        if s.take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        u32()  # version
        n_tensors, n_kv = u64(), u64()
        kv = {}
        for _ in range(n_kv):
            k = string()
            t = u32()
            v = value(t, keep=not k.startswith("tokenizer.ggml.") or k == "tokenizer.ggml.pre")
            if v is not None:
                kv[k] = v
        params = 0
        for _ in range(n_tensors):
            string()
            dims = [u64() for _ in range(u32())]
            u32(), u64()  # type, offset
            n = 1
            for d in dims:
                n *= d
            params += n
        return kv, params
    finally:
        s.close()


def main():
    out, problems = [], []
    for mid, name, maker, licence, about, repo, extra in MODELS:
        try:
            tree = get_json(f"{HF}/api/models/{repo}/tree/main")
        except Exception as e:
            problems.append(f"{repo}: {e}")
            continue
        files = []
        for q in QUANTS:
            cand = [f for f in tree if f["type"] == "file" and f["path"].lower().endswith(f"{q.lower()}.gguf") and "/" not in f["path"] and "-of-" not in f["path"] and "lfs" in f]
            if cand:
                f = min(cand, key=lambda f: len(f["path"]))
                files.append({"quant": q, "file": f["path"], "size": f["lfs"]["size"], "sha256": f["lfs"]["oid"]})
        if not files:
            problems.append(f"{repo}: no single-file {'/'.join(QUANTS)} GGUF")
            continue
        smallest = min(files, key=lambda f: f["size"])
        try:
            kv, params = gguf_header(f"{HF}/{repo}/resolve/main/{smallest['file']}")
        except Exception as e:
            problems.append(f"{repo}: reading the header: {e}")
            continue
        arch = kv.get("general.architecture", "")
        a = lambda k, d=None: kv.get(f"{arch}.{k}", d)
        tmpl = kv.get("tokenizer.chat_template", "") or ""
        caps = set(extra)
        if "tools" in tmpl:
            caps.add("tools")
        if "<think>" in tmpl or "enable_thinking" in tmpl:
            caps.add("thinking")
        if a("pooling_type") is not None or arch in ("bert", "nomic-bert"):
            caps.add("embedding")
        n_head, n_head_kv, n_embd = a("attention.head_count", 0), a("attention.head_count_kv"), a("embedding_length", 0)
        head_kv = n_head_kv if isinstance(n_head_kv, int) else n_head
        out.append({
            "id": mid, "name": name, "maker": maker, "license": licence, "about": about, "repo": repo,
            "arch": arch, "tokenizer_pre": kv.get("tokenizer.ggml.pre", "gpt2"),
            "params_b": round(params / 1e9, 2), "n_layers": a("block_count", 0), "hidden_dim": n_embd,
            "kv_ratio": round(head_kv / n_head, 4) if n_head else 1.0,
            "context": a("context_length", 0), "caps": sorted(caps), "files": files,
        })
        print(f"{mid}: {arch}/{out[-1]['tokenizer_pre']} {out[-1]['params_b']}B, {out[-1]['n_layers']} layers, caps {sorted(caps)}, {[f['quant'] for f in files]}", file=sys.stderr)
    dest = Path(__file__).resolve().parent.parent / "crates/clamor-node/src/catalog.json"
    dest.write_text(json.dumps({"generated_by": "tools/build-catalog.py", "models": out}, indent=1) + "\n")
    print(f"wrote {dest}: {len(out)} models", file=sys.stderr)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
