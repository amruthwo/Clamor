//! People in a swarm, their roles, and how they sign in to the dashboard.
//!
//! * A **person** has devices (their keys) and a role: the **owner** (who started the swarm),
//!   **admins** (appointed by the owner), and **members**. Everyone who joins becomes a person;
//!   "Add my device" links another device to the same person.
//! * **Who is asking** (see portal.rs): the coordinator machine itself is the owner (whoever can
//!   run clamor-node there controls the swarm anyway); the app on a member device signs its
//!   requests with that device's key; a browser has a session from a sign-in code approved on a
//!   signed-in device, or from the person's PIN.
use crate::auth::{hmac, random_bytes};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Member,
    Admin,
    Owner,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub id: String,
    pub name: String,
    pub role: Role,
    /// PIN for signing in from a browser: "salt:hash" (PBKDF2-HMAC-SHA256), hex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin: Option<String>,
    pub created_unix: u64,
    /// How the model's replies appear: "stream" (as they are written; the default) or "whole".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_style: Option<String>,
    /// Keys for the OpenAI-compatible API (/v1), which act as this person. Only hashes are kept.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub api_keys: Vec<ApiKey>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    /// What the person called it ("Zed on my laptop").
    pub name: String,
    /// SHA-256 of the key, hex.
    pub hash: String,
    /// The key's first characters, to tell keys apart ("clm_3f9a…").
    pub hint: String,
    pub created_unix: u64,
}

/// A new API key: (the key, shown once; its record). Keys are 32 random bytes: hashing (no
/// salt, no stretching) is enough, unlike PINs.
pub fn new_api_key(name: &str) -> (String, ApiKey) {
    let key = format!("clm_{}", hex::encode(crate::auth::random_bytes::<32>()));
    let rec = ApiKey { id: hex::encode(crate::auth::random_bytes::<6>()), name: name.chars().take(60).collect(), hash: hash_api_key(&key), hint: key[..8].to_string(), created_unix: clamor_proto::unix_now() };
    (key, rec)
}

pub fn hash_api_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(key.as_bytes()))
}

pub const OWNER_ID: &str = "owner";

pub fn new_person_id() -> String {
    hex::encode(random_bytes::<6>())
}

// ---------------------------------------------------------------- PINs

const PIN_ROUNDS: u32 = 100_000;

/// PBKDF2-HMAC-SHA256, one 32-byte block.
fn pbkdf2(pin: &[u8], salt: &[u8], rounds: u32) -> [u8; 32] {
    let mut msg = salt.to_vec();
    msg.extend_from_slice(&1u32.to_be_bytes());
    let mut u = hmac(pin, &msg);
    let mut out = u;
    for _ in 1..rounds {
        u = hmac(pin, &u);
        for (o, x) in out.iter_mut().zip(u.iter()) {
            *o ^= x;
        }
    }
    out
}

pub fn valid_pin(pin: &str) -> Result<(), String> {
    if !(4..=12).contains(&pin.len()) || !pin.chars().all(|c| c.is_ascii_digit()) {
        return Err("a PIN is 4 to 12 digits".into());
    }
    Ok(())
}

pub fn hash_pin(pin: &str) -> String {
    let salt = random_bytes::<16>();
    format!("{}:{}", hex::encode(salt), hex::encode(pbkdf2(pin.as_bytes(), &salt, PIN_ROUNDS)))
}

pub fn check_pin(stored: &str, pin: &str) -> bool {
    let Some((salt, hash)) = stored.split_once(':') else { return false };
    let (Ok(salt), Ok(hash)) = (hex::decode(salt), hex::decode(hash)) else { return false };
    crate::auth::ct_eq(&pbkdf2(pin.as_bytes(), &salt, PIN_ROUNDS), &hash)
}

/// Wrong PINs: after this many in `PIN_WINDOW_SECS`, a person's PIN sign-in is locked until the
/// window passes. A 4-digit PIN cannot be guessed at 5 tries per quarter hour.
pub const PIN_TRIES: u32 = 5;
pub const PIN_WINDOW_SECS: u64 = 900;

#[derive(Default)]
pub struct PinGuard {
    failures: HashMap<String, (u32, u64)>,
}

impl PinGuard {
    pub fn locked(&self, person: &str, now: u64) -> bool {
        self.failures.get(person).is_some_and(|(n, since)| *n >= PIN_TRIES && now.saturating_sub(*since) < PIN_WINDOW_SECS)
    }
    pub fn failed(&mut self, person: &str, now: u64) {
        let e = self.failures.entry(person.into()).or_insert((0, now));
        if now.saturating_sub(e.1) >= PIN_WINDOW_SECS {
            *e = (0, now);
        }
        e.0 += 1;
    }
    pub fn succeeded(&mut self, person: &str) {
        self.failures.remove(person);
    }
}

// ---------------------------------------------------------------- browser sessions

pub const SESSION_DAYS: u64 = 30;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Session {
    person: String,
    expires_unix: u64,
}

/// Browser sessions, by the SHA-256 of their token (the file never holds a usable token).
#[derive(Default, Serialize, Deserialize)]
pub struct Sessions {
    by_hash: HashMap<String, Session>,
}

fn token_hash(token: &str) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(token.as_bytes()))
}

impl Sessions {
    fn path(config_dir: &Path, swarm_id: &str) -> std::path::PathBuf {
        config_dir.join("swarms").join(format!("sessions-{swarm_id}.json"))
    }
    pub fn load(config_dir: &Path, swarm_id: &str) -> Sessions {
        std::fs::read(Self::path(config_dir, swarm_id)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }
    fn save(&self, config_dir: &Path, swarm_id: &str) {
        if let Err(e) = crate::auth::write_private(&Self::path(config_dir, swarm_id), &serde_json::to_vec(self).unwrap()) {
            eprintln!("could not save sessions: {e:#}");
        }
    }
    /// A new session for `person`; returns the token (for the cookie).
    pub fn create(&mut self, person: &str, now: u64, config_dir: &Path, swarm_id: &str) -> String {
        let token = hex::encode(random_bytes::<32>());
        self.by_hash.retain(|_, s| s.expires_unix > now);
        self.by_hash.insert(token_hash(&token), Session { person: person.into(), expires_unix: now + SESSION_DAYS * 86_400 });
        self.save(config_dir, swarm_id);
        token
    }
    pub fn person(&self, token: &str, now: u64) -> Option<String> {
        self.by_hash.get(&token_hash(token)).filter(|s| s.expires_unix > now).map(|s| s.person.clone())
    }
    pub fn end(&mut self, token: &str, config_dir: &Path, swarm_id: &str) {
        self.by_hash.remove(&token_hash(token));
        self.save(config_dir, swarm_id);
    }
    /// Sign a person out everywhere (removed, or PIN changed).
    pub fn end_all(&mut self, person: &str, config_dir: &Path, swarm_id: &str) {
        self.by_hash.retain(|_, s| s.person != person);
        self.save(config_dir, swarm_id);
    }
}

// ---------------------------------------------------------------- sign-in by code

/// A browser waiting for someone signed in to approve it (TV-style).
pub struct PendingSignIn {
    pub id: String,
    pub code: String,
    pub created_unix: u64,
    /// Who approved it (the browser becomes that person), or None while waiting.
    pub approved_by: Option<String>,
    /// From where (shown to the approver).
    pub from: String,
    /// Someone signed in pressed "Not me": the browser is told, and the code no longer works.
    pub declined: bool,
}

pub const SIGNIN_SECS: u64 = 300;

pub fn six_digits() -> String {
    let n = u32::from_le_bytes(random_bytes::<4>()) % 1_000_000;
    format!("{:03}-{:03}", n / 1000, n % 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbkdf2_matches_rfc_7914_vector() {
        // RFC 7914 section 11: PBKDF2-HMAC-SHA256("passwd", "salt", 1), first 32 bytes.
        assert_eq!(hex::encode(pbkdf2(b"passwd", b"salt", 1)), "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc");
    }

    #[test]
    fn pins_hash_check_and_lock() {
        let h = hash_pin("4821");
        assert!(check_pin(&h, "4821"));
        assert!(!check_pin(&h, "4822"));
        assert_ne!(h, hash_pin("4821"), "salted");
        assert!(valid_pin("12").is_err() && valid_pin("12ab").is_err() && valid_pin("1234").is_ok());
        let mut g = PinGuard::default();
        for _ in 0..PIN_TRIES {
            assert!(!g.locked("p", 1000));
            g.failed("p", 1000);
        }
        assert!(g.locked("p", 1000));
        assert!(!g.locked("p", 1000 + PIN_WINDOW_SECS), "unlocks after the window");
    }

    #[test]
    fn sessions_expire_and_end() {
        let dir = std::env::temp_dir().join(format!("clamor-sessions-{}", std::process::id()));
        let mut s = Sessions::default();
        let t = s.create("bea", 1000, &dir, "x");
        assert_eq!(s.person(&t, 1001).as_deref(), Some("bea"));
        assert_eq!(s.person(&t, 1000 + SESSION_DAYS * 86_400), None, "expired");
        assert_eq!(Sessions::load(&dir, "x").person(&t, 1001).as_deref(), Some("bea"), "saved");
        assert!(!std::fs::read_to_string(dir.join("swarms/sessions-x.json")).unwrap().contains(&t), "only hashes on disk");
        s.end_all("bea", &dir, "x");
        assert_eq!(s.person(&t, 1001), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
