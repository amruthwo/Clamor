//! An OpenAI-compatible API on the dashboard's ports (/v1), so tools made for OpenAI's API can
//! use the swarm: coding assistants, chat front ends, agent frameworks. Point them at
//! `http(s)://<coordinator>:7421/v1` (or, on a member device, its own `http://127.0.0.1:7421/v1`)
//! with an API key from Settings -> API keys.
//!
//! Supported: `GET /v1/models`, `POST /v1/chat/completions` (streaming or not; `max_tokens`,
//! `stop`, `stream_options.include_usage`). The swarm runs one model, so every request goes to
//! it whatever `model` says; sampling (temperature and so on) is the running model's. Tool
//! calls are not supported yet: `tools` is ignored and the model answers in text.
use crate::coordinator::Shared;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use clamor_engine_ggml::{encode_chat, ChatMessage};
use clamor_proto::unix_now;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

/// An error in OpenAI's shape.
pub fn error(status: StatusCode, message: &str, code: &str) -> Response {
    let kind = match status.as_u16() {
        401 => "authentication_error",
        403 => "permission_error",
        429 | 503 => "server_error",
        400..=499 => "invalid_request_error",
        _ => "server_error",
    };
    (status, Json(json!({ "error": { "message": message, "type": kind, "code": code } }))).into_response()
}

/// The running model's name, as the API calls it.
fn model_name(shared: &Shared) -> Option<String> {
    let sw = shared.lock().unwrap();
    sw.pipeline.as_ref().map(|p| p.model.trim_end_matches(".gguf").to_string())
}

pub async fn models(State(shared): State<Shared>) -> Json<Value> {
    let data: Vec<Value> = model_name(&shared).into_iter().map(|id| json!({ "id": id, "object": "model", "created": unix_now(), "owned_by": "clamor" })).collect();
    Json(json!({ "object": "list", "data": data }))
}

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<Value>,
}

/// A message's text: a string, or the text parts of a list of parts.
fn text_of(v: &Option<Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// OpenAI's roles in the model's chat template: "developer" is a system prompt; tool results
/// read as user messages (tool calls are not supported yet).
fn role_of(r: &str) -> &'static str {
    match r {
        "system" | "developer" => "system",
        "assistant" => "assistant",
        _ => "user",
    }
}

/// Where the reply stops: at the first stop string, if any.
fn cut(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()
}

/// How much of `text` is safe to send while more may follow: not a trailing part that could
/// still become a stop string, nor an unfinished character.
fn safe_len(text: &str, stops: &[String]) -> usize {
    let hold = stops.iter().map(|s| s.len().saturating_sub(1)).max().unwrap_or(0);
    let mut end = text.len().saturating_sub(hold);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let t = &text[..end];
    t.strip_suffix('\u{FFFD}').map(str::len).unwrap_or(end)
}

pub async fn chat_completions(State(shared): State<Shared>, body: axum::body::Bytes) -> Response {
    let req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("invalid request: {e}"), "invalid_request"),
    };
    if req.n.unwrap_or(1) != 1 {
        return error(StatusCode::BAD_REQUEST, "only n = 1 is supported", "unsupported");
    }
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages is empty", "invalid_request");
    }
    let (head, _epoch, n_ctx, tok) = match crate::portal::ready_pipeline(&shared) {
        Ok(x) => x,
        Err((s, m)) => return error(s, &format!("{m} (start a model in Clamor: Models)"), "model_not_ready"),
    };
    let Some(tok) = tok else { return error(StatusCode::NOT_IMPLEMENTED, "this model's tokenizer is not supported yet", "unsupported") };
    let model = model_name(&shared).or(req.model.clone()).unwrap_or_default();
    let msgs: Vec<ChatMessage> = req.messages.iter().map(|m| ChatMessage { role: role_of(&m.role).into(), content: text_of(&m.content) }).collect();
    let prompt = match encode_chat(&tok, &msgs) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string(), "invalid_request"),
    };
    let room = n_ctx.saturating_sub(prompt.len() as u32);
    if room == 0 {
        return error(StatusCode::BAD_REQUEST, &format!("the messages are {} tokens; the context holds {n_ctx}", prompt.len()), "context_length_exceeded");
    }
    let max_new = req.max_completion_tokens.or(req.max_tokens).unwrap_or(room).clamp(1, room);
    let stops: Vec<String> = match req.stop {
        Some(Stop::One(s)) => vec![s],
        Some(Stop::Many(v)) => v.into_iter().take(4).collect(),
        None => Vec::new(),
    };
    let id = format!("chatcmpl-{}", hex::encode(crate::auth::random_bytes::<12>()));
    let created = unix_now();
    let prompt_tokens = prompt.len();
    let stop = Arc::new(tokio::sync::Notify::new());

    if !req.stream {
        let (tok2, stops2, stop2) = (tok.clone(), stops.clone(), stop.clone());
        let r = crate::generate::run_with(head, prompt, max_new, Some(stop.clone()), move |t| {
            if !stops2.is_empty() && cut(&tok2.decode(t, true), &stops2).is_some() {
                stop2.notify_one();
            }
        })
        .await;
        if let Some(e) = r.error {
            return error(StatusCode::INTERNAL_SERVER_ERROR, &e, "generation_failed");
        }
        let mut text = tok.decode(&r.tokens, true);
        let hit = cut(&text, &stops);
        if let Some(at) = hit {
            text.truncate(at);
        }
        let finish = if r.done == "max_tokens" && hit.is_none() { "length" } else { "stop" };
        return Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": finish }],
            "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": r.tokens.len(), "total_tokens": prompt_tokens + r.tokens.len() },
        }))
        .into_response();
    }

    // Streaming: server-sent events, one chunk per piece of text, then [DONE].
    let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    let chunk = {
        let (id, model) = (id.clone(), model.clone());
        move |delta: Value, finish: Option<&str>| format!("data: {}\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }))
    };
    let _ = tx.send(Ok(chunk(json!({ "role": "assistant", "content": "" }), None)));
    tokio::spawn(async move {
        let sent = Arc::new(std::sync::Mutex::new(0usize));
        let (tok2, stops2, stop2, tx2, sent2, chunk2) = (tok.clone(), stops.clone(), stop.clone(), tx.clone(), sent.clone(), chunk.clone());
        let r = crate::generate::run_with(head, prompt, max_new, Some(stop.clone()), move |t| {
            let text = tok2.decode(t, true);
            let end = match cut(&text, &stops2) {
                Some(at) => {
                    stop2.notify_one();
                    at
                }
                None => safe_len(&text, &stops2),
            };
            let mut s = sent2.lock().unwrap();
            if end > *s {
                let _ = tx2.send(Ok(chunk2(json!({ "content": &text[*s..end] }), None)));
                *s = end;
            }
        })
        .await;
        if let Some(e) = r.error {
            let _ = tx.send(Ok(format!("data: {}\n\n", json!({ "error": { "message": e, "type": "server_error", "code": "generation_failed" } }))));
        } else {
            // What was held back (it did not become a stop string).
            let text = tok.decode(&r.tokens, true);
            let hit = cut(&text, &stops);
            let end = hit.unwrap_or(text.len());
            let s = *sent.lock().unwrap();
            if end > s {
                let _ = tx.send(Ok(chunk(json!({ "content": &text[s..end] }), None)));
            }
            let finish = if r.done == "max_tokens" && hit.is_none() { "length" } else { "stop" };
            let _ = tx.send(Ok(chunk(json!({}), Some(finish))));
            if include_usage {
                let _ = tx.send(Ok(format!("data: {}\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": r.tokens.len(), "total_tokens": prompt_tokens + r.tokens.len() } }))));
            }
        }
        let _ = tx.send(Ok("data: [DONE]\n\n".into()));
    });
    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings_and_held_back_text() {
        let stops = vec!["\nUser:".to_string()];
        assert_eq!(cut("Hello\nUser: hi", &stops), Some(5));
        assert_eq!(cut("Hello", &stops), None);
        // "Hello\nUs" could still become "\nUser:": hold back the last 5 bytes.
        assert_eq!(safe_len("Hello\nUs", &stops), 3);
        assert_eq!(safe_len("Hello", &[]), 5);
        assert_eq!(safe_len("café", &[]), "café".len());
        assert_eq!(safe_len("ab\u{FFFD}", &[]), 2, "an unfinished character waits");
    }

    #[test]
    fn message_content_and_roles() {
        assert_eq!(text_of(&Some(json!("hi"))), "hi");
        assert_eq!(text_of(&Some(json!([{ "type": "text", "text": "a" }, { "type": "image_url", "image_url": {} }, { "type": "text", "text": "b" }]))), "a\nb");
        assert_eq!(text_of(&None), "");
        assert_eq!((role_of("developer"), role_of("tool"), role_of("assistant")), ("system", "user", "assistant"));
    }
}
