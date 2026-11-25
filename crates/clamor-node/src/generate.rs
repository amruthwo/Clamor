//! Writing a model's reply to a conversation, shared by the coordinator's dashboard (group and
//! personal chats) and a device's own dashboard (private chats, whose ends run on the device).
use clamor_engine_ggml::{encode_chat, Tokenizer};
use clamor_transport::stage::{DoneReason, Event, HeadHandle};
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) struct CancelOnDrop {
    head: HeadHandle,
    req: u64,
    done: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.done {
            self.head.cancel(self.req);
        }
    }
}

pub struct Run {
    pub tokens: Vec<u32>,
    pub done: &'static str,
    pub error: Option<String>,
    pub ms: u64,
    pub first_token_ms: Option<u64>,
    pub decode_tok_s: Option<f64>,
}

/// Submit a prompt to the head and collect the reply.
pub async fn run(head: HeadHandle, prompt: Vec<u32>, max_new: u32) -> Run {
    run_with(head, prompt, max_new, None, |_| {}).await
}

/// `run`, calling `on_token` with the tokens so far after each one, and stopping early (the
/// head cancels the request; what was written is kept) when `stop` is notified.
pub async fn run_with(head: HeadHandle, prompt: Vec<u32>, max_new: u32, stop: Option<Arc<tokio::sync::Notify>>, mut on_token: impl FnMut(&[u32])) -> Run {
    let t0 = Instant::now();
    let (req, mut rx) = head.submit(prompt, max_new);
    let mut guard = CancelOnDrop { head: head.clone(), req, done: false };
    let mut tokens = Vec::new();
    let mut first_ms = None;
    let mut stopping = false;
    let reason = loop {
        let stop_now = async {
            match (&stop, stopping) {
                (Some(s), false) => s.notified().await,
                _ => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = stop_now => {
                stopping = true;
                head.cancel(req);
            }
            ev = tokio::time::timeout(Duration::from_secs(600), rx.recv()) => match ev {
                Err(_) => break DoneReason::Failed("timed out after 600 s without a token".into()),
                Ok(None) => break DoneReason::Failed("pipeline stopped".into()),
                Ok(Some(Event::Token { id, .. })) => {
                    first_ms.get_or_insert(t0.elapsed().as_millis() as u64);
                    tokens.push(id);
                    on_token(&tokens);
                }
                Ok(Some(Event::Done(r))) => break r,
            }
        }
    };
    guard.done = !matches!(reason, DoneReason::Failed(_));
    let ms = t0.elapsed().as_millis() as u64;
    let (done, error) = match reason {
        DoneReason::MaxTokens => ("max_tokens", None),
        DoneReason::Eos => ("eos", None),
        DoneReason::Cancelled => ("cancelled", None),
        DoneReason::Failed(e) => ("failed", Some(e)),
    };
    // Decode rate excludes the first token, which includes the whole prompt (prefill).
    let decode_tok_s = match (first_ms, tokens.len()) {
        (Some(f), n) if n > 1 && ms > f => Some((n - 1) as f64 / ((ms - f) as f64 / 1000.0)),
        _ => None,
    };
    Run { tokens, done, error, ms, first_token_ms: first_ms, decode_tok_s }
}

/// A reply being written: what it says so far, and the finished message at the end.
#[derive(Clone, Default, Serialize)]
pub struct Progress {
    pub text: String,
    pub tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<crate::chats::ChatMsg>,
}

/// A conversation the model is answering right now.
pub struct Generation {
    /// Who asked (they, or an admin, may stop it).
    pub asker: String,
    pub stop: Arc<tokio::sync::Notify>,
    pub progress: tokio::sync::watch::Receiver<Progress>,
}

/// Build the prompt for a conversation: the whole conversation, the oldest messages dropped
/// until it fits with room for a reply (and as a last resort the group's system note).
pub fn chat_prompt(tok: &Tokenizer, conv: &crate::chats::Conversation, n_ctx: u32) -> Result<Vec<u32>, String> {
    let mut msgs = crate::chats::model_messages(conv);
    let mut keep_system = usize::from(msgs.first().is_some_and(|m| m.role == "system"));
    let budget = n_ctx.saturating_sub(256.min(n_ctx / 2)) as usize;
    let prompt = loop {
        match encode_chat(tok, &msgs) {
            Ok(p) if p.len() <= budget => break p,
            Ok(_) if msgs.len() > keep_system + 1 => {
                msgs.remove(keep_system);
            }
            Ok(_) if keep_system == 1 => {
                msgs.remove(0);
                keep_system = 0;
            }
            Ok(p) => break p,
            Err(e) => return Err(e.to_string()),
        }
    };
    if prompt.len() as u32 >= n_ctx {
        return Err(format!("that message alone is longer than the model's context ({n_ctx} tokens)"));
    }
    Ok(prompt)
}

/// Write the reply to `conv` (whose prompt is `prompt`) in its own task: progress goes to `tx`
/// as it is written, `finish` gets the finished message (to save), then `tx` gets it too.
#[allow(clippy::too_many_arguments)]
pub fn spawn_reply(
    head: HeadHandle,
    tok: Arc<Tokenizer>,
    n_ctx: u32,
    prompt: Vec<u32>,
    conv: crate::chats::Conversation,
    stop: Arc<tokio::sync::Notify>,
    tx: tokio::sync::watch::Sender<Progress>,
    finish: impl FnOnce(&crate::chats::ChatMsg) + Send + 'static,
) {
    tokio::spawn(async move {
        let room = n_ctx - prompt.len() as u32;
        let (tx2, tok2, conv2) = (tx.clone(), tok.clone(), conv.clone());
        let r = run_with(head, prompt, 1024.min(room), Some(stop), move |tokens| {
            let text = crate::chats::tidy_reply(&conv2, &tok2.decode(tokens, true));
            tx2.send_replace(Progress { text, tokens: tokens.len() as u32, done: None });
        })
        .await;
        let text = crate::chats::tidy_reply(&conv, &tok.decode(&r.tokens, true));
        let at = clamor_proto::unix_now();
        let msg = match &r.error {
            Some(e) => crate::chats::ChatMsg { role: "assistant".into(), content: text, by: None, at, tok_s: None, error: Some(e.clone()), stopped: false },
            None => crate::chats::ChatMsg { role: "assistant".into(), content: text, by: None, at, tok_s: r.decode_tok_s.map(|s| s as f32), error: None, stopped: r.done == "cancelled" },
        };
        finish(&msg);
        tx.send_replace(Progress { text: msg.content.clone(), tokens: r.tokens.len() as u32, done: Some(msg) });
    });
}

/// A reply that could not even start: the finished message says why.
pub fn failed(reason: String) -> crate::chats::ChatMsg {
    crate::chats::ChatMsg { role: "assistant".into(), content: String::new(), by: None, at: clamor_proto::unix_now(), tok_s: None, error: Some(reason), stopped: false }
}

/// The asker's response: one JSON object per line, `{"text", "tokens"}` while the reply is
/// written, then `{"done": message}`. The reply goes on (and is saved) if the asker leaves.
pub fn progress_body(mut rx: tokio::sync::watch::Receiver<Progress>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let (btx, brx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::convert::Infallible>>(16);
    tokio::spawn(async move {
        loop {
            let p = rx.borrow_and_update().clone();
            let finished = p.done.is_some();
            let mut line = serde_json::to_vec(&p).unwrap();
            line.push(b'\n');
            if (!p.text.is_empty() || finished) && btx.send(Ok(line.into())).await.is_err() {
                return;
            }
            if finished || rx.changed().await.is_err() {
                return;
            }
        }
    });
    let body = axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(brx));
    ([(axum::http::header::CONTENT_TYPE, "application/x-ndjson"), (axum::http::header::CACHE_CONTROL, "no-cache")], body).into_response()
}
