#!/usr/bin/env python3
"""Render a model's chat template the way Hugging Face's apply_chat_template does, for tests.

Usage: chat_ref.py HF_REPO CONVERSATIONS.json OUT.json

Fetches tokenizer_config.json from the Hub, renders each conversation with jinja2
(ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, raise_exception, strftime_now,
add_generation_prompt=True) and writes {template, bos_token, eos_token, rendered: [...]}.
"""
import json, sys
from datetime import datetime
from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment

repo, convs_path, out_path = sys.argv[1:4]
cfg = json.load(open(hf_hub_download(repo, "tokenizer_config.json"), encoding="utf-8"))
template = cfg["chat_template"]
if isinstance(template, list):  # some repos store named templates
    template = next(t["template"] for t in template if t["name"] == "default")
def tok_str(t):
    return t["content"] if isinstance(t, dict) else (t or "")
bos, eos = tok_str(cfg.get("bos_token")), tok_str(cfg.get("eos_token"))

def raise_exception(msg):
    raise Exception(msg)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda f: datetime.now().strftime(f)
t = env.from_string(template)
rendered = [t.render(messages=c, add_generation_prompt=True, bos_token=bos, eos_token=eos) for c in json.load(open(convs_path, encoding="utf-8"))]
json.dump({"template": template, "bos_token": bos, "eos_token": eos, "rendered": rendered}, open(out_path, "w", encoding="utf-8"))
print(f"{repo}: rendered {len(rendered)} conversations")
