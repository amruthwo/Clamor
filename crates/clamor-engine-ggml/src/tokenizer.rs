//! Byte-level BPE tokenizer (the GPT-2 family: Llama 3, Qwen2, SmolLM, StarCoder, ...), built
//! from a GGUF's `tokenizer.ggml.*` metadata.
//!
//! Pipeline: split out special tokens -> pre-tokenize with the model's regexes -> map bytes to
//! GPT-2's printable byte alphabet -> apply BPE merges by rank -> look up ids.
//!
//! Pre-tokenizer rules are per model family (`tokenizer.ggml.pre`). Only families checked
//! against Hugging Face's reference tokenizer are listed; anything else is refused, because a
//! wrong guess still "works" but silently produces different token ids.
//!
//! Control tokens (`<|im_start|>` and friends) are recognised only when the caller asks
//! (`parse_special`), which chat templates do and user text must not: otherwise a user could
//! type `<|im_end|>` and forge a turn boundary.
use crate::gguf::GgufFile;
use crate::{Error, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::Mutex;

/// `tokenizer.ggml.token_type` values (llama.cpp's convention).
pub mod token_type {
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Whether this tokenizer handles a model's `tokenizer.ggml.pre` (its pre-tokenizer).
pub fn supports_pre(pre: &str) -> bool {
    pre_rules(pre).is_some()
}

/// (regexes applied in sequence, ignore_merges) for a `tokenizer.ggml.pre` value.
fn pre_rules(pre: &str) -> Option<(&'static [&'static str], bool)> {
    Some(match pre {
        "gpt2" => (&[GPT2], false),
        // Digits split one by one, then GPT-2's rules.
        "smollm" | "starcoder" => (&[r"\p{N}", GPT2], false),
        // Llama 3 also takes a whole word straight from the vocabulary when it is there.
        "llama-bpe" | "llama3" => (&[LLAMA3], true),
        "qwen2" => (&[QWEN2], false),
        _ => return None,
    })
}

/// GPT-2's reversible map from bytes to printable characters, so every byte sequence has a
/// text form that BPE can work on.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut enc = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        enc[b as usize] = if printable {
            b as char
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    let dec = enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
    (enc, dec)
}

pub struct Tokenizer {
    /// Token text in the byte alphabet, by id.
    tokens: Vec<String>,
    types: Vec<i32>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    /// Literal-matched tokens, longest first: (text, id, is_control).
    specials: Vec<(String, u32, bool)>,
    pre: Vec<Regex>,
    ignore_merges: bool,
    enc: [char; 256],
    dec: HashMap<char, u8>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /// End-of-turn token for chat (falls back to eos).
    pub eot: Option<u32>,
    pub add_bos: bool,
    /// The model's chat template (Jinja), if the GGUF has one.
    pub chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or_default();
        if model != "gpt2" {
            return Err(Error::Gguf(format!("tokenizer model {model:?} is not supported yet (only byte-level BPE, \"gpt2\")")));
        }
        let tokens = g.get_arr_str("tokenizer.ggml.tokens").ok_or_else(|| Error::Gguf("no tokenizer.ggml.tokens".into()))?;
        // No merges is a valid (if inefficient) vocabulary: every byte is its own token.
        let merges = g.get_arr_str("tokenizer.ggml.merges").unwrap_or_default();
        let types = g.get_arr_i32("tokenizer.ggml.token_type").unwrap_or_else(|| vec![token_type::NORMAL; tokens.len()]);
        let pre = g.get_str("tokenizer.ggml.pre").unwrap_or_else(|| "gpt2".into());
        let mut t = Self::from_parts(tokens, &merges, types, &pre)?;
        t.bos = g.get_u32("tokenizer.ggml.bos_token_id");
        t.eos = g.get_u32("tokenizer.ggml.eos_token_id");
        t.eot = g.get_u32("tokenizer.ggml.eot_token_id").or(t.eos);
        t.add_bos = g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
        t.chat_template = g.get_str("tokenizer.chat_template");
        Ok(t)
    }

    /// Build from raw parts: token texts (byte alphabet) by id, merges as `"left right"`
    /// strings in rank order, token types, and the pre-tokenizer name.
    pub fn from_parts(tokens: Vec<String>, merges: &[String], types: Vec<i32>, pre: &str) -> Result<Self> {
        let (rules, ignore_merges) = pre_rules(pre).ok_or_else(|| Error::Gguf(format!("pre-tokenizer {pre:?} is not supported yet")))?;
        if types.len() != tokens.len() {
            return Err(Error::Gguf(format!("{} token types for {} tokens", types.len(), tokens.len())));
        }
        let pre = rules.iter().map(|r| Regex::new(r).expect("built-in pre-tokenizer regex")).collect();
        let ids = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut ranks = HashMap::with_capacity(merges.len());
        for (r, m) in merges.iter().enumerate() {
            let (a, b) = m.split_once(' ').ok_or_else(|| Error::Gguf(format!("bad merge {m:?}")))?;
            ranks.entry((a.to_string(), b.to_string())).or_insert(r as u32);
        }
        let mut specials: Vec<(String, u32, bool)> = types
            .iter()
            .enumerate()
            .filter(|(_, &ty)| ty == token_type::CONTROL || ty == token_type::USER_DEFINED)
            .filter(|(i, _)| !tokens[*i].is_empty())
            .map(|(i, &ty)| (tokens[i].clone(), i as u32, ty == token_type::CONTROL))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let (enc, dec) = byte_alphabet();
        Ok(Self { tokens, types, ids, ranks, specials, pre, ignore_merges, enc, dec, cache: Mutex::new(HashMap::new()), bos: None, eos: None, eot: None, add_bos: false, chat_template: None })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.types.get(id as usize).is_some_and(|&t| t == token_type::CONTROL)
    }

    /// Text to token ids. `parse_special`: recognise control tokens written as text (for chat
    /// templates, never for user input). User-defined added tokens are always recognised, as
    /// in llama.cpp and Hugging Face. BOS is not added; chat templates include it themselves.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        // Fragments: Ok(text to tokenize) or Err(special token id).
        let mut frags: Vec<std::result::Result<&str, u32>> = vec![Ok(text)];
        for (s, id, control) in &self.specials {
            if *control && !parse_special {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len());
            for f in frags {
                match f {
                    Ok(t) if t.contains(s.as_str()) => {
                        let mut rest = t;
                        while let Some(i) = rest.find(s.as_str()) {
                            if i > 0 {
                                next.push(Ok(&rest[..i]));
                            }
                            next.push(Err(*id));
                            rest = &rest[i + s.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Ok(rest));
                        }
                    }
                    other => next.push(other),
                }
            }
            frags = next;
        }
        let mut out = Vec::new();
        for f in frags {
            match f {
                Err(id) => out.push(id),
                Ok(t) => {
                    for piece in self.pre_tokenize(t) {
                        self.bpe(piece, &mut out);
                    }
                }
            }
        }
        out
    }

    /// Split by each pre-tokenizer regex in turn; text between matches stays as its own piece.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = vec![text];
        for re in &self.pre {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for p in pieces {
                let mut last = 0;
                for m in re.find_iter(p).flatten() {
                    if m.start() > last {
                        next.push(&p[last..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(m.as_str());
                    }
                    last = m.end();
                }
                if last < p.len() {
                    next.push(&p[last..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let word: String = piece.bytes().map(|b| self.enc[b as usize]).collect();
        if let Some(ids) = self.cache.lock().unwrap().get(&word) {
            out.extend_from_slice(ids);
            return;
        }
        let ids = match self.ids.get(&word) {
            Some(&id) if self.ignore_merges => vec![id],
            _ => self.merge(&word),
        };
        out.extend_from_slice(&ids);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() > 50_000 {
            cache.clear();
        }
        cache.insert(word, ids);
    }

    fn merge(&self, word: &str) -> Vec<u32> {
        let mut syms: Vec<String> = word.chars().map(String::from).collect();
        while syms.len() > 1 {
            // Lowest-ranked adjacent pair.
            let best = (0..syms.len() - 1).filter_map(|i| self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())).map(|&r| (r, i))).min();
            let Some((_, i)) = best else { break };
            let (a, b) = (syms[i].clone(), syms[i + 1].clone());
            // Merge every occurrence of this pair, left to right.
            let mut merged = Vec::with_capacity(syms.len());
            let mut k = 0;
            while k < syms.len() {
                if k + 1 < syms.len() && syms[k] == a && syms[k + 1] == b {
                    merged.push(format!("{a}{b}"));
                    k += 2;
                } else {
                    merged.push(std::mem::take(&mut syms[k]));
                    k += 1;
                }
            }
            syms = merged;
        }
        syms.iter()
            .flat_map(|s| match self.ids.get(s) {
                Some(&id) => vec![id],
                // Every single byte character is a token in byte-level vocabularies; this only
                // guards against a broken vocabulary.
                None => s.chars().filter_map(|c| self.ids.get(&c.to_string()).copied()).collect(),
            })
            .collect()
    }

    /// The raw bytes a token stands for (control tokens: their text).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.types[id as usize] == token_type::CONTROL || self.types[id as usize] == token_type::USER_DEFINED {
            return t.as_bytes().to_vec();
        }
        let mut out = Vec::with_capacity(t.len());
        for c in t.chars() {
            match self.dec.get(&c) {
                Some(&b) => out.push(b),
                None => out.extend_from_slice(c.to_string().as_bytes()),
            }
        }
        out
    }

    /// Token ids to text. `skip_control`: leave out control tokens such as `<|im_end|>`.
    pub fn decode(&self, ids: &[u32], skip_control: bool) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&id| !(skip_control && self.is_control(id))).flat_map(|&id| self.token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Turns a stream of token ids into text, holding back bytes until they form whole UTF-8
/// characters (one emoji is often several tokens).
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32, skip_control: bool) -> String {
        if skip_control && tok.is_control(id) {
            return String::new();
        }
        self.pending.extend(tok.token_bytes(id));
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) if e.error_len().is_none() => {
                // Incomplete character at the end: emit the valid prefix, keep the rest.
                let n = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.pending[..n]).into_owned();
                self.pending.drain(..n);
                s
            }
            Err(_) => {
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                s
            }
        }
    }

    /// Whatever is left at the end of a stream.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (enc, dec) = byte_alphabet();
        assert_eq!(dec.len(), 256);
        for b in 0..=255u8 {
            assert_eq!(dec[&enc[b as usize]], b);
        }
        assert_eq!(enc[b' ' as usize], 'Ġ');
        assert_eq!(enc[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn unknown_pre_tokenizers_are_refused() {
        assert!(Tokenizer::from_parts(vec!["a".into()], &[], vec![1], "mystery").is_err());
    }
}
