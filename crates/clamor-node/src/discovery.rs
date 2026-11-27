//! Finding a swarm on the local network without typing an address (mDNS / DNS-SD).
//!
//! A coordinator listening on all interfaces advertises `_clamor._tcp.local.` with its swarm's
//! id and name. `clamor-node join` without an address looks for those: a swarm this device has
//! joined before is found by its id (so a new IP address from the router does not matter), and
//! a first join picks the only swarm on the network, or the one named with `--swarm`.
//!
//! What is advertised is public on the network: the swarm's name and id. The passphrase and keys
//! are not, and finding a swarm does not let anyone join it.
use anyhow::{anyhow, Result};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::{Duration, Instant};

pub const SERVICE: &str = "_clamor._tcp.local.";

/// A swarm seen on the network.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub swarm_id: String,
    pub name: String,
    /// Every address the coordinator advertised, IPv4 first.
    pub addrs: Vec<SocketAddr>,
}

impl Found {
    /// The first advertised address that accepts a connection (a machine can advertise
    /// addresses others cannot reach, such as a container bridge).
    pub fn reachable(&self) -> Option<SocketAddr> {
        self.addrs.iter().copied().find(|a| TcpStream::connect_timeout(a, Duration::from_millis(800)).is_ok())
    }
}

/// Advertise this coordinator until the process exits.
pub fn advertise(swarm_id: &str, name: &str, port: u16) -> Result<()> {
    let daemon = ServiceDaemon::new()?;
    let host = clamor_muster::probe::default_node_id().0;
    let host: String = host.split('.').next().unwrap_or("").chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
    let host = format!("{}.local.", if host.is_empty() { "clamor".into() } else { host });
    let instance = format!("{name} {}", &swarm_id[..swarm_id.len().min(8)]);
    let props = [("id", swarm_id), ("name", name), ("v", "3")];
    let info = ServiceInfo::new(SERVICE, &instance, &host, "", port, &props[..])?.enable_addr_auto();
    daemon.register(info)?;
    // The daemon runs on its own thread; keep its handle for the life of the process.
    std::mem::forget(daemon);
    Ok(())
}

/// Swarms advertised on the network, collected for `wait`.
/// An address another device can connect to: not a loopback (it would mean the device itself),
/// and not an IPv6 link-local one (it needs the network interface, which an address alone does
/// not carry: connecting fails with "Invalid argument").
fn usable(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => !v4.is_loopback() && !v4.is_unspecified(),
        IpAddr::V6(v6) => !v6.is_loopback() && !v6.is_unspecified() && (v6.segments()[0] & 0xffc0) != 0xfe80,
    }
}

pub fn browse(wait: Duration) -> Result<Vec<Found>> {
    let daemon = ServiceDaemon::new()?;
    let rx = daemon.browse(SERVICE)?;
    let deadline = Instant::now() + wait;
    let mut found: Vec<Found> = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        let Ok(ev) = rx.recv_timeout(left) else { break };
        if let ServiceEvent::ServiceResolved(s) = ev {
            let (Some(id), Some(name)) = (s.get_property_val_str("id"), s.get_property_val_str("name")) else { continue };
            let ips: Vec<IpAddr> = s.addresses.iter().map(|a| a.to_ip_addr()).filter(usable).collect();
            let addrs: Vec<SocketAddr> = ips.into_iter().map(|ip| SocketAddr::new(ip, s.port)).collect();
            match found.iter_mut().find(|f| f.swarm_id == id) {
                Some(f) => {
                    for a in addrs {
                        if !f.addrs.contains(&a) {
                            f.addrs.push(a);
                        }
                    }
                }
                None => found.push(Found { swarm_id: id.into(), name: name.into(), addrs }),
            }
        }
    }
    let _ = daemon.shutdown();
    // Addresses arrive over several events: order the merged list, IPv4 first.
    for f in &mut found {
        f.addrs.sort_by_key(|a| (!a.is_ipv4(), *a));
    }
    found.retain(|f| !f.addrs.is_empty());
    Ok(found)
}

/// Which swarm a `join` without an address should go to: a remembered one (most recent first)
/// that is on the network; otherwise the one named `want`, or, for a device that remembers no
/// swarm at all, the only swarm on the network. A device that has joined swarms before never
/// wanders into an unknown one by itself.
pub fn choose<'a>(found: &'a [Found], remembered: &[String], want: Option<&str>) -> Result<&'a Found> {
    let named = |f: &&Found| want.is_none_or(|w| f.name == w);
    for id in remembered {
        if let Some(f) = found.iter().filter(named).find(|f| &f.swarm_id == id) {
            return Ok(f);
        }
    }
    if want.is_none() && !remembered.is_empty() {
        return Err(anyhow!("none of the swarms this device joined is on the network"));
    }
    let candidates: Vec<&Found> = found.iter().filter(named).collect();
    match (candidates.len(), want) {
        (1, _) => Ok(candidates[0]),
        (0, Some(w)) => Err(anyhow!("no swarm called '{w}' found on the network")),
        (0, None) => Err(anyhow!("no swarm found on the network")),
        _ => {
            let names: Vec<&str> = candidates.iter().map(|f| f.name.as_str()).collect();
            Err(anyhow!("several swarms on the network ({}): pick one with --swarm <name>", names.join(", ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(id: &str, name: &str) -> Found {
        Found { swarm_id: id.into(), name: name.into(), addrs: vec![] }
    }

    #[test]
    fn choosing_a_swarm() {
        let net = [f("a", "home"), f("b", "friends")];
        // A remembered swarm wins, most recent first.
        assert_eq!(choose(&net, &["b".into(), "a".into()], None).unwrap().swarm_id, "b");
        assert_eq!(choose(&net, &["zz".into(), "a".into()], None).unwrap().swarm_id, "a");
        // None remembered: ambiguous unless named.
        assert!(choose(&net, &[], None).unwrap_err().to_string().contains("--swarm"));
        assert_eq!(choose(&net, &[], Some("friends")).unwrap().swarm_id, "b");
        assert!(choose(&net, &[], Some("work")).is_err());
        // Only one: take it.
        assert_eq!(choose(&net[..1], &[], None).unwrap().swarm_id, "a");
        assert!(choose(&[], &[], None).is_err());
        // A device that knows other swarms does not join an unknown one unless named.
        assert!(choose(&net[..1], &["zz".into()], None).is_err());
        assert_eq!(choose(&net[..1], &["zz".into()], Some("home")).unwrap().swarm_id, "a");
        // --swarm also picks among remembered ones.
        assert_eq!(choose(&net, &["a".into(), "b".into()], Some("friends")).unwrap().swarm_id, "b");
    }
}
