//! Chat prompts: render a model's own chat template (Jinja, from the GGUF) and tokenize it.
//!
//! Rendering matches Hugging Face's `apply_chat_template` environment: `trim_blocks` and
//! `lstrip_blocks` on, Python string methods available, `raise_exception` and
//! `strftime_now` provided. Models without a template get ChatML.
//!
//! Safety: a rendered prompt is tokenized with control tokens recognised, since the template
//! writes them. Message text must not be: if any message contains text that looks like a
//! control token (say a user types `<|im_end|>`), those messages are tokenized as plain text
//! and spliced in, so the text cannot forge a turn boundary.
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use minijinja::{Environment, ErrorKind, Value};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

/// Used when a model has no template of its own.
pub const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// Today's date via a tiny subset of `strftime`, for templates that print it (Llama 3.x). Local
/// time, like Hugging Face's (Python's) `strftime_now`: in the evening west of Greenwich the UTC
/// date is already tomorrow.
fn strftime_now(fmt: String) -> String {
    let now = chrono::Local::now();
    strftime_at(fmt, now.timestamp() + i64::from(now.offset().local_minus_utc()))
}

/// `strftime_now` for `secs` seconds since 1970-01-01 in local time.
fn strftime_at(fmt: String, secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let month = MONTHS[(m - 1) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('B') => out.push_str(month),
            Some('b') => out.push_str(&month[..3]),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().expect("valid syntax config"));
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> std::result::Result<Value, minijinja::Error> { Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg)) });
    env.add_function("strftime_now", strftime_now);
    env
}

/// Render `template` (or ChatML) for `messages`, ending with the assistant's turn opening.
pub fn render(template: Option<&str>, messages: &[ChatMessage], bos: &str, eos: &str) -> Result<String> {
    let env = environment();
    let ctx = minijinja::context! {
        messages => Value::from(minijinja::value::Serde(messages)),
        add_generation_prompt => true,
        bos_token => bos,
        eos_token => eos,
    };
    env.render_str(template.unwrap_or(CHATML), ctx).map_err(|e| Error::Ggml(format!("chat template: {e:#}")))
}

/// The prompt's token ids, ready to send to the head stage.
pub fn encode_chat(tok: &Tokenizer, messages: &[ChatMessage]) -> Result<Vec<u32>> {
    let name = |id: Option<u32>| id.map(|i| tok.decode(&[i], false)).unwrap_or_default();
    let (bos, eos) = (name(tok.bos), name(tok.eos));
    let template = tok.chat_template.as_deref();
    let forged = |m: &ChatMessage| tok.encode(&m.content, true) != tok.encode(&m.content, false);
    if !messages.iter().any(forged) {
        return Ok(tok.encode(&render(template, messages, &bos, &eos)?, true));
    }
    // Render with sentinels in place of the message texts, then tokenize the template parts
    // with control tokens and the message texts without.
    let sentinel = |i: usize| format!("\u{E000}clamor-message-{i}\u{E001}");
    let masked: Vec<ChatMessage> = messages.iter().enumerate().map(|(i, m)| ChatMessage { role: m.role.clone(), content: sentinel(i) }).collect();
    let mut rest = render(template, &masked, &bos, &eos)?;
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let s = sentinel(i);
        let Some(at) = rest.find(&s) else {
            return Err(Error::Ggml("chat template dropped or altered a message; cannot tokenize it safely".into()));
        };
        out.extend(tok.encode(&rest[..at], true));
        out.extend(tok.encode(&m.content, false));
        rest = rest[at + s.len()..].to_string();
    }
    out.extend(tok.encode(&rest, true));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback() {
        let s = render(None, &[ChatMessage::new("user", "Hi")], "", "").unwrap();
        assert_eq!(s, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn template_errors_are_reported() {
        let t = "{{ raise_exception('roles must alternate') }}";
        let e = render(Some(t), &[], "", "").unwrap_err().to_string();
        assert!(e.contains("roles must alternate"), "{e}");
    }

    #[test]
    fn strftime_formats_known_dates() {
        // 2026-11-16 00:30 local.
        assert_eq!(strftime_at("%d %b %Y".into(), 1_794_789_000), "16 Nov 2026");
        assert_eq!(strftime_at("%Y-%m-%d".into(), 951_782_400), "2000-02-29");
    }

    #[test]
    fn strftime_formats_a_plausible_date() {
        let s = strftime_now("%d %b %Y".into());
        let parts: Vec<&str> = s.split(' ').collect();
        assert_eq!(parts.len(), 3, "{s}");
        assert!(parts[2].parse::<i32>().unwrap() >= 2024, "{s}");
        assert_eq!(strftime_now("100%%".into()), "100%");
    }
}
