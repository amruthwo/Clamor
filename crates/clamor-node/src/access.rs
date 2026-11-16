//! Who may use the dashboard, and for what (see people.rs for the model).
//!
//! Every request is identified before it reaches a handler, in this order:
//! 1. signed by an approved device's key (`x-clamor-auth` over "METHOD path"), only on the
//!    encrypted dashboard: the app on a member device, through its local proxy (dashproxy.rs);
//! 2. a browser session cookie (sign-in code approved by a signed-in person, or a PIN);
//! 3. from the coordinator machine itself (loopback): the owner.
//!
//! Then the route's minimum role is checked. Handlers that need finer rules (settings of one's
//! own device) look at the [`Caller`].
use crate::coordinator::{Shared, Swarm};
use crate::people::{Role, OWNER_ID};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::net::SocketAddr;

/// Which listener a request came in on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    Plain,
    Tls,
}

#[derive(Clone, Debug)]
pub struct Caller {
    pub person: String,
    pub name: String,
    pub role: Role,
    /// How they were recognised: "coordinator", "device" or "session".
    pub how: &'static str,
}

pub const SESSION_COOKIE: &str = "clamor_session";

pub fn session_token(headers: &HeaderMap) -> Option<String> {
    let cookies = headers.get(axum::http::header::COOKIE)?.to_str().ok()?;
    cookies.split(';').filter_map(|c| c.trim().split_once('=')).find(|(k, _)| *k == SESSION_COOKIE).map(|(_, v)| v.to_string())
}

/// Test hook: treat loopback like any other address, so tests can act as members.
fn loopback_is_owner() -> bool {
    std::env::var("CLAMOR_TEST_NO_LOOPBACK_OWNER").as_deref() != Ok("1")
}

pub fn identify(sw: &Swarm, peer: SocketAddr, via: Via, method: &Method, path_q: &str, headers: &HeaderMap) -> Option<Caller> {
    let profile = &sw.auth.profile;
    let as_caller = |id: &str, how| profile.person(id).map(|p| Caller { person: p.id.clone(), name: p.name.clone(), role: p.role, how });
    // A device's signature (only over TLS: the body is not signed, TLS protects it).
    if via == Via::Tls {
        if let Some(h) = headers.get(crate::auth::SHARE_AUTH_HEADER).and_then(|v| v.to_str().ok()) {
            let signed = format!("{method} {path_q}");
            return crate::auth::verify_share_request(h, &signed, |k| profile.approved.iter().any(|d| d.key == k))
                .ok()
                .and_then(|key| profile.person_of_key(&key).map(|p| p.id.clone()))
                .and_then(|id| as_caller(&id, "device"));
        }
    }
    // A browser that signed in.
    if let Some(person) = session_token(headers).and_then(|t| sw.auth.sessions.person(&t, clamor_proto::unix_now())) {
        return as_caller(&person, "session");
    }
    // The coordinator machine itself.
    if peer.ip().is_loopback() && loopback_is_owner() {
        let owner = profile.people.iter().find(|p| p.role == Role::Owner).map(|p| p.id.clone()).unwrap_or_else(|| OWNER_ID.into());
        return as_caller(&owner, "coordinator");
    }
    None
}

/// The least role a route needs; None: open to anyone (the page itself, and signing in).
pub fn required(method: &Method, path: &str) -> Option<Role> {
    let p: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    let m = method.as_str();
    match (m, p.as_slice()) {
        ("GET", [""]) | ("GET", ["favicon.ico"]) => None,
        ("POST", ["api", "login", "start"]) | ("GET", ["api", "login", "poll"]) | ("POST", ["api", "login", "pin"]) => None,
        ("POST", ["api", "people", _, "role"]) => Some(Role::Owner),
        ("POST", ["api", "muster"])
        | ("POST", ["api", "nodes", _, "drain"])
        | ("POST", ["api", "nodes", _, "remove"])
        | ("POST", ["api", "nodes", _, "person"])
        | ("POST", ["api", "people", _, "remove"])
        | ("POST", ["api", "pipeline"])
        | ("DELETE", ["api", "pipeline"]) => Some(Role::Admin),
        _ => Some(Role::Member),
    }
}

pub async fn layer(State(shared): State<Shared>, mut req: Request, next: Next) -> Response {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0)
        .or_else(|| req.extensions().get::<ConnectInfo<crate::portal::TlsPeer>>().map(|c| c.0 .0))
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 0)));
    let via = req.extensions().get::<Via>().copied().unwrap_or(Via::Plain);
    let Some(need) = required(req.method(), req.uri().path()) else {
        req.extensions_mut().insert(peer);
        return next.run(req).await;
    };
    let path_q = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or_default();
    let caller = {
        let sw = shared.lock().unwrap();
        identify(&sw, peer, via, req.method(), &path_q, req.headers())
    };
    let Some(caller) = caller else {
        return (StatusCode::UNAUTHORIZED, "sign in first").into_response();
    };
    if caller.role < need {
        let what = if need == Role::Owner { "the swarm's owner" } else { "an admin" };
        return (StatusCode::FORBIDDEN, format!("only {what} can do that")).into_response();
    }
    req.extensions_mut().insert(caller);
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_need_the_right_roles() {
        let get = Method::GET;
        let post = Method::POST;
        assert_eq!(required(&get, "/"), None);
        assert_eq!(required(&post, "/api/login/start"), None);
        assert_eq!(required(&get, "/api/roster"), Some(Role::Member));
        assert_eq!(required(&post, "/api/chat"), Some(Role::Member));
        assert_eq!(required(&post, "/api/nodes/x/policy"), Some(Role::Member), "own device: checked in the handler");
        assert_eq!(required(&post, "/api/nodes/x/remove"), Some(Role::Admin));
        assert_eq!(required(&post, "/api/pipeline"), Some(Role::Admin));
        assert_eq!(required(&get, "/api/pipeline"), Some(Role::Member));
        assert_eq!(required(&post, "/api/people/p/role"), Some(Role::Owner));
        assert_eq!(required(&get, "/api/anything-new"), Some(Role::Member), "unknown routes are not open");
        assert!(Role::Owner > Role::Admin && Role::Admin > Role::Member);
    }

    #[test]
    fn session_cookie_is_found_among_others() {
        let mut h = HeaderMap::new();
        h.insert(axum::http::header::COOKIE, "a=1; clamor_session=abc; b=2".parse().unwrap());
        assert_eq!(session_token(&h).as_deref(), Some("abc"));
    }
}
