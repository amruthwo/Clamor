#!/usr/bin/env python3
"""Reference tokenization with Hugging Face `tokenizers`, for the Rust tokenizer tests.

Usage: tok_ref.py HF_REPO_OR_TOKENIZER_JSON CORPUS.json OUT.json

Writes the vocabulary (by id), merges ("left right", in rank order), token types (llama.cpp's
GGUF convention: special added tokens are CONTROL=3, other added tokens USER_DEFINED=4,
the rest NORMAL=1) and, for every corpus string, the ids with special tokens recognised
("special") and with special tokens treated as plain text ("plain").
"""
import json, os, sys
from tokenizers import Tokenizer

src, corpus_path, out_path = sys.argv[1:4]
tk = Tokenizer.from_file(src) if os.path.exists(src) else Tokenizer.from_pretrained(src)
spec = json.loads(tk.to_str())
vocab = spec["model"]["vocab"]
added = {a["id"]: a for a in spec["added_tokens"]}
n = max(max(vocab.values()), max(added, default=-1)) + 1
tokens = [""] * n
for t, i in vocab.items():
    tokens[i] = t
types = [1] * n
for i, a in added.items():
    tokens[i] = a["content"]
    types[i] = 3 if a["special"] else 4
merges = [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]]

corpus = json.load(open(corpus_path, encoding="utf-8"))
cases = []
for text in corpus:
    tk.encode_special_tokens = False
    special = tk.encode(text, add_special_tokens=False).ids
    tk.encode_special_tokens = True
    plain = tk.encode(text, add_special_tokens=False).ids
    cases.append({"text": text, "special": special, "plain": plain})
json.dump({"tokens": tokens, "merges": merges, "types": types, "cases": cases}, open(out_path, "w", encoding="utf-8"))
print(f"{src}: {n} tokens, {len(merges)} merges, {len(cases)} cases")
