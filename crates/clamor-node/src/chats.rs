//! Conversations, kept by the coordinator so they follow a person between their devices.
//!
//! * **Personal** conversations belong to one person; nobody else in the swarm can open them
//!   (the swarm's owner, who runs the coordinator, could read the files: private chats that
//!   even the owner cannot read come later).
//! * The **group** conversation is shared by everyone in the swarm; each message says who wrote
//!   it, and the model sees the names too.
//!
//! One JSON file per conversation, under `<config>/swarms/chats-<swarm id>/`.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const GROUP: &str = "group";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMsg {
    /// "user" or "assistant".
    pub role: String,
    pub content: String,
    /// Who wrote it (user messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    pub at: u64,
    /// For replies: how fast the model wrote it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tok_s: Option<f32>,
    /// For replies that failed: why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    /// The person it belongs to; None for the group.
    pub owner: Option<String>,
    pub messages: Vec<ChatMsg>,
    pub updated: u64,
}

pub struct Chats {
    dir: PathBuf,
    convs: HashMap<String, Conversation>,
}

/// Messages kept per conversation (older ones are dropped).
const KEEP: usize = 500;

impl Chats {
    pub fn load(config_dir: &Path, swarm_id: &str) -> Chats {
        let dir = config_dir.join("swarms").join(format!("chats-{swarm_id}"));
        let mut convs = HashMap::new();
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            if let Some(c) = std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<Conversation>(&b).ok()) {
                convs.insert(c.id.clone(), c);
            }
        }
        convs.entry(GROUP.into()).or_insert_with(|| Conversation { id: GROUP.into(), title: "Group".into(), owner: None, messages: Vec::new(), updated: 0 });
        Chats { dir, convs }
    }

    fn save(&self, id: &str) {
        let Some(c) = self.convs.get(id) else { return };
        let r = crate::auth::write_private(&self.dir.join(format!("{id}.json")), &serde_json::to_vec(c).unwrap());
        if let Err(e) = r {
            eprintln!("could not save conversation {id}: {e:#}");
        }
    }

    /// May `person` open conversation `id`?
    pub fn can_open(&self, id: &str, person: &str) -> bool {
        self.convs.get(id).is_some_and(|c| c.owner.as_deref().is_none_or(|o| o == person))
    }

    pub fn get(&self, id: &str) -> Option<&Conversation> {
        self.convs.get(id)
    }

    /// The person's own conversations (newest first), without the messages.
    pub fn list(&self, person: &str) -> Vec<serde_json::Value> {
        let mut v: Vec<&Conversation> = self.convs.values().filter(|c| c.owner.as_deref() == Some(person)).collect();
        v.sort_by_key(|c| std::cmp::Reverse(c.updated));
        v.iter().map(|c| serde_json::json!({ "id": c.id, "title": c.title, "updated": c.updated, "count": c.messages.len() })).collect()
    }

    pub fn create(&mut self, person: &str, now: u64) -> String {
        let id = hex::encode(crate::auth::random_bytes::<8>());
        self.convs.insert(id.clone(), Conversation { id: id.clone(), title: "New chat".into(), owner: Some(person.into()), messages: Vec::new(), updated: now });
        self.save(&id);
        id
    }

    pub fn append(&mut self, id: &str, msg: ChatMsg) {
        let Some(c) = self.convs.get_mut(id) else { return };
        // A personal chat is named after its first question.
        if c.owner.is_some() && c.messages.is_empty() && msg.role == "user" {
            let t: String = msg.content.split_whitespace().collect::<Vec<_>>().join(" ");
            c.title = if t.chars().count() > 48 { format!("{}…", t.chars().take(47).collect::<String>()) } else { t };
        }
        c.updated = msg.at;
        c.messages.push(msg);
        if c.messages.len() > KEEP {
            let drop = c.messages.len() - KEEP;
            c.messages.drain(..drop);
        }
        self.save(id);
    }

    pub fn delete(&mut self, id: &str) {
        if id != GROUP && self.convs.remove(id).is_some() {
            let _ = std::fs::remove_file(self.dir.join(format!("{id}.json")));
        }
    }

    /// Forget everything a removed person had.
    pub fn remove_person(&mut self, person: &str) {
        let ids: Vec<String> = self.convs.values().filter(|c| c.owner.as_deref() == Some(person)).map(|c| c.id.clone()).collect();
        for id in ids {
            self.delete(&id);
        }
    }
}

/// What the model sees for a conversation: in the group, every message is prefixed with its
/// author's name, and a short system message explains that.
pub fn model_messages(c: &Conversation) -> Vec<clamor_engine_ggml::ChatMessage> {
    let mut out = Vec::new();
    if c.owner.is_none() {
        out.push(clamor_engine_ggml::ChatMessage { role: "system".into(), content: "This is a group chat. Each person's message starts with their name and a colon. Answer the latest message directly; do not start your reply with anyone's name.".into() });
    }
    for m in c.messages.iter().filter(|m| m.error.is_none()) {
        let content = match (&m.by, c.owner.is_none(), m.role.as_str()) {
            (Some(by), true, "user") => format!("{by}: {}", m.content),
            _ => m.content.clone(),
        };
        out.push(clamor_engine_ggml::ChatMessage { role: m.role.clone(), content });
    }
    out
}

/// Small models sometimes imitate the "Name: " format in a group chat: drop a leading name of
/// anyone in the conversation from a reply.
pub fn tidy_reply(c: &Conversation, reply: &str) -> String {
    let r = reply.trim();
    if c.owner.is_none() {
        for name in c.messages.iter().filter_map(|m| m.by.as_deref()) {
            if let Some(rest) = r.strip_prefix(name).and_then(|x| x.strip_prefix(':')) {
                return rest.trim().to_string();
            }
            let lower = r.to_lowercase();
            if lower.starts_with(&format!("{}:", name.to_lowercase())) {
                return r[name.len() + 1..].trim().to_string();
            }
        }
    }
    r.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str, by: Option<&str>) -> ChatMsg {
        ChatMsg { role: role.into(), content: content.into(), by: by.map(Into::into), at: 1, tok_s: None, error: None }
    }

    #[test]
    fn personal_chats_are_private_and_kept() {
        let dir = std::env::temp_dir().join(format!("clamor-chats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut c = Chats::load(&dir, "s");
        let id = c.create("bea", 5);
        c.append(&id, msg("user", "What is   the capital of France?", Some("Bea")));
        assert!(c.can_open(&id, "bea") && !c.can_open(&id, "cal"), "personal");
        assert!(c.can_open(GROUP, "cal"), "the group is everyone's");
        assert_eq!(c.list("bea")[0]["title"], "What is the capital of France?");
        assert!(c.list("cal").is_empty());
        let again = Chats::load(&dir, "s");
        assert_eq!(again.get(&id).unwrap().messages.len(), 1, "saved");
        c.remove_person("bea");
        assert!(c.get(&id).is_none() && Chats::load(&dir, "s").get(&id).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_model_sees_names_in_the_group_only() {
        let mut g = Conversation { id: GROUP.into(), title: "Group".into(), owner: None, messages: vec![], updated: 0 };
        g.messages.push(msg("user", "hi", Some("Bea")));
        let m = model_messages(&g);
        assert_eq!((m[0].role.as_str(), m[1].content.as_str()), ("system", "Bea: hi"));
        assert_eq!(tidy_reply(&g, "Bea: Paris."), "Paris.");
        assert_eq!(tidy_reply(&g, "bea: Paris."), "Paris.");
        assert_eq!(tidy_reply(&g, "Paris: a city."), "Paris: a city.", "only names of people in the chat");
        g.owner = Some("bea".into());
        assert_eq!(model_messages(&g)[0].content, "hi");
        assert_eq!(tidy_reply(&g, "Bea: hi"), "Bea: hi", "personal chats are left alone");
    }
}
