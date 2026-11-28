//! Model handover: a device leaving the swarm (on its own, or removed by an admin) may hold the
//! swarm's only copy of a model. Before it goes, the model is copied to a device with room (the
//! coordinator if it can: downloads go there anyway), checked to be the same model, and only
//! then is the device told to leave. If no device has room, the caller is told which models
//! would be lost and can leave anyway. If a copy fails, the device stays and the leave is
//! called off (nothing is lost); trying again starts over.
use crate::coordinator::{Shared, Swarm};
use clamor_proto::*;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Clone, Debug, Serialize)]
pub struct Handover {
    pub model_id: String,
    pub name: String,
    pub size: u64,
    pub from: String,
    pub to: String,
    pub done: u64,
    /// "copying", "done" or "failed".
    pub state: String,
    pub error: Option<String>,
    pub started_unix: u64,
}

/// Margin kept free on the receiving device.
const SPARE: u64 = 1 << 30;
/// Give up on a copy that has not finished after this long.
const LIMIT: Duration = Duration::from_secs(3 * 3600);

/// The models only `id` has among the devices in the swarm right now.
pub fn only_copies(sw: &Swarm, id: &NodeId) -> Vec<ModelInfo> {
    let Some(e) = sw.nodes.get(id) else { return Vec::new() };
    e.models
        .iter()
        .filter(|m| !sw.nodes.iter().any(|(other, o)| other != id && (o.tx.is_some() || *other == sw.coordinator) && o.models.iter().any(|x| x.id == m.id)))
        .cloned()
        .collect()
}

/// Whether `id` can receive a handover: the coordinator, or a connected device that understands it.
fn can_receive(sw: &Swarm, id: &NodeId) -> bool {
    *id == sw.coordinator || sw.nodes.get(id).is_some_and(|e| e.tx.is_some() && e.build.as_ref().is_some_and(|b| b.features.iter().any(|f| f == "handover")))
}

/// Where each only copy of `from` would go: Ok(plan) or Err(names of models no device has room for).
pub fn plan(sw: &Swarm, from: &NodeId) -> Result<Vec<(ModelInfo, NodeId)>, Vec<String>> {
    let mut room: HashMap<NodeId, u64> = sw.nodes.iter().filter(|(id, _)| *id != from && can_receive(sw, id)).map(|(id, e)| (id.clone(), e.report.info.disk_free_bytes.saturating_sub(SPARE))).collect();
    let (mut out, mut lost) = (Vec::new(), Vec::new());
    for m in only_copies(sw, from) {
        // The coordinator first (downloads live there), else the device with the most room.
        let coordinator_fits = room.get(&sw.coordinator).is_some_and(|r| *r >= m.size);
        let to = if coordinator_fits { Some(sw.coordinator.clone()) } else { room.iter().filter(|(_, r)| **r >= m.size).max_by_key(|(_, r)| **r).map(|(id, _)| id.clone()) };
        match to {
            Some(to) => {
                *room.get_mut(&to).unwrap() -= m.size;
                out.push((m, to));
            }
            None => lost.push(m.name.clone()),
        }
    }
    if lost.is_empty() {
        Ok(out)
    } else {
        Err(lost)
    }
}

/// Start copying `from`'s only copies to the planned devices. When they are all in place, the
/// device is removed (`then_remove`, after `drain_after`); if one fails, the leave is called off.
pub fn start(shared: &Shared, sw: &mut Swarm, from: &NodeId, copies: Vec<(ModelInfo, NodeId)>, drain_after: Duration) -> Vec<Handover> {
    let now = unix_now();
    let mut started = Vec::new();
    for (m, to) in copies {
        let Some(source) = crate::pipeline::share_url(sw, from, &to) else {
            started.push(Handover { model_id: m.id.clone(), name: m.name.clone(), size: m.size, from: from.0.clone(), to: to.0.clone(), done: 0, state: "failed".into(), error: Some("cannot reach the leaving device's model sharing".into()), started_unix: now });
            continue;
        };
        let h = Handover { model_id: m.id.clone(), name: m.name.clone(), size: m.size, from: from.0.clone(), to: to.0.clone(), done: 0, state: "copying".into(), error: None, started_unix: now };
        println!("handover: copying {} from {from} to {to} before {from} leaves", m.name);
        if to == sw.coordinator {
            // The coordinator fetches it itself.
            let (dir, key, shared2, id) = (sw.engine.models_dir.clone(), sw.engine.share_key.clone(), shared.clone(), m.id.clone());
            let name = m.name.clone();
            std::thread::spawn(move || {
                let r = crate::share::fetch_file(&source, &name, &id, key.as_ref(), &dir, |done, _| set(&shared2, &id, |h| h.done = done));
                let found = crate::share::scan(&dir);
                let mut sw = shared2.lock().unwrap();
                let me = sw.coordinator.clone();
                if let Some(e) = sw.nodes.get_mut(&me) {
                    e.models = found;
                }
                finish(&mut sw, &id, r.err().map(|e| format!("{e:#}")));
            });
        } else if let Some(tx) = sw.nodes.get(&to).and_then(|e| e.tx.clone()) {
            let _ = tx.send(Message::FetchModel { name: m.name.clone(), model_id: m.id.clone(), source });
        }
        started.push(h);
    }
    sw.handovers.retain(|h| h.from != from.0);
    sw.handovers.extend(started.iter().cloned());
    // Let the device go once every copy is in place.
    let (shared, from) = (shared.clone(), from.clone());
    tokio::spawn(async move {
        let t0 = std::time::Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let mut sw = shared.lock().unwrap();
            let mine: Vec<&Handover> = sw.handovers.iter().filter(|h| h.from == from.0).collect();
            if mine.iter().any(|h| h.state == "failed") {
                println!("handover: a copy failed; {from} stays in the swarm");
                return;
            }
            if t0.elapsed() > LIMIT {
                for h in sw.handovers.iter_mut().filter(|h| h.from == from.0 && h.state == "copying") {
                    h.state = "failed".into();
                    h.error = Some("took too long".into());
                }
                return;
            }
            if !mine.is_empty() && mine.iter().all(|h| h.state == "done") {
                println!("handover: {from}'s models are safe elsewhere; it can leave");
                let _ = crate::portal::remove_device(&mut sw, from.clone(), drain_after);
                return;
            }
            if !sw.nodes.contains_key(&from) {
                return; // removed meanwhile
            }
        }
    });
    started
}

fn set(shared: &Shared, model_id: &str, f: impl FnOnce(&mut Handover)) {
    let mut sw = shared.lock().unwrap();
    if let Some(h) = sw.handovers.iter_mut().rev().find(|h| h.model_id == model_id && h.state == "copying") {
        f(h);
    }
}

/// A copy finished (`error` None) or failed.
pub fn finish(sw: &mut Swarm, model_id: &str, error: Option<String>) {
    if let Some(h) = sw.handovers.iter_mut().rev().find(|h| h.model_id == model_id && h.state == "copying") {
        match error {
            None => {
                h.state = "done".into();
                h.done = h.size;
                println!("handover: {} is now on {}", h.name, h.to);
            }
            Some(e) => {
                eprintln!("handover: copying {} to {} failed: {e}", h.name, h.to);
                h.state = "failed".into();
                h.error = Some(e);
            }
        }
    }
}

/// A device reported progress on a copy it was asked to make.
pub fn progress(sw: &mut Swarm, to: &NodeId, model_id: &str, done: u64, finished: bool, error: Option<String>) {
    if let Some(h) = sw.handovers.iter_mut().rev().find(|h| h.model_id == model_id && h.to == to.0 && h.state == "copying") {
        h.done = done;
    }
    if finished || error.is_some() {
        finish(sw, model_id, if finished { None } else { error });
    }
}
