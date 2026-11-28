//! Chat templates rendered in Rust (minijinja) against Python jinja2 configured like Hugging
//! Face's apply_chat_template (tools/chat_ref.py), on each supported family's real template.
//!
//! Needs python3 with jinja2 + huggingface_hub and network access (skips without, unless
//! CLAMOR_REQUIRE_HF=1, set in CI).
use clamor_engine_ggml::*;
use serde_json::Value;

const REPOS: &[&str] = &["HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"];

fn conversations() -> Vec<Vec<ChatMessage>> {
    let m = ChatMessage::new;
    vec![
        vec![m("user", "Hi there!")],
        vec![m("system", "You are terse."), m("user", "What is 2+2?")],
        vec![m("user", "Name a colour."), m("assistant", "Blue."), m("user", "Another?")],
        vec![m("system", "  Leading and trailing spaces  \n"), m("user", "\n  Héllo 😀 中文\t\n")],
        vec![m("user", "Code:\n```rust\nfn main() {}\n```\nExplain it.")],
    ]
}

fn reference(repo: &str) -> Option<Value> {
    let dir = std::env::temp_dir().join(format!("clamor-chat-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let convs = dir.join("convs.json");
    std::fs::write(&convs, serde_json::to_string(&conversations()).unwrap()).unwrap();
    let out = dir.join(format!("{}.json", repo.replace('/', "_")));
    let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/chat_ref.py");
    let ok = std::process::Command::new("python3")
        .args([tool, repo, convs.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .map(|o| {
            if !o.status.success() {
                eprintln!("{}", String::from_utf8_lossy(&o.stderr));
            }
            o.status.success()
        })
        .unwrap_or(false);
    if !ok {
        if std::env::var("CLAMOR_REQUIRE_HF").as_deref() == Ok("1") {
            panic!("tools/chat_ref.py failed for {repo} (CLAMOR_REQUIRE_HF=1)");
        }
        eprintln!("skipping {repo}: python3 + jinja2 + huggingface_hub + network needed");
        return None;
    }
    Some(serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap())
}

#[test]
fn templates_render_like_hugging_face() {
    for repo in REPOS {
        let Some(r) = reference(repo) else { continue };
        let template = r["template"].as_str().unwrap();
        let (bos, eos) = (r["bos_token"].as_str().unwrap(), r["eos_token"].as_str().unwrap());
        for (i, (conv, want)) in conversations().iter().zip(r["rendered"].as_array().unwrap()).enumerate() {
            let got = render_chat(Some(template), conv, bos, eos).unwrap_or_else(|e| panic!("{repo} #{i}: {e}"));
            assert_eq!(got, want.as_str().unwrap(), "{repo}, conversation {i}");
        }
    }
}
