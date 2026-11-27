//! The apps run a node in-process (`clamor_node::run_args`) and stop it with `shutdown()`
//! instead of Ctrl-C (Android: the app is the process). It must stop promptly, free its ports,
//! and be startable again.
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

fn port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

#[test]
fn a_node_started_in_process_stops_on_shutdown() {
    let dir = std::env::temp_dir().join(format!("clamor-embedded-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    for round in 0..2 {
        let (ctrl, portal, tls, data) = (port(), port(), port(), 20_000 + (std::process::id() % 5_000) as u16 + round * 100);
        let args: Vec<String> = [
            "coordinator", "--swarm", "embedded", "--passphrase", "in-process", "--bind", &format!("127.0.0.1:{ctrl}"), "--portal", &format!("127.0.0.1:{portal}"), "--portal-tls", &format!("127.0.0.1:{tls}"),
            "--data-port", &data.to_string(), "--models", &dir.join("models").display().to_string(), "--config-dir", &dir.join("config").display().to_string(), "--quick-bench", "--no-update-check", "--no-mdns",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let node = std::thread::spawn(move || clamor_node::run_args(args));
        let t0 = Instant::now();
        while TcpStream::connect(("127.0.0.1", portal)).is_err() {
            assert!(t0.elapsed() < Duration::from_secs(60), "the portal never opened (round {round})");
            assert!(!node.is_finished(), "the node stopped by itself");
            std::thread::sleep(Duration::from_millis(100));
        }
        clamor_node::shutdown();
        let t1 = Instant::now();
        while !node.is_finished() {
            assert!(t1.elapsed() < Duration::from_secs(30), "the node did not stop on shutdown() (round {round})");
            std::thread::sleep(Duration::from_millis(100));
        }
        node.join().unwrap().expect("a clean stop");
        // Its control port is free again.
        TcpListener::bind(("127.0.0.1", ctrl)).expect("the control port was released");
    }
    assert!(clamor_node::run_args(vec!["no-such-command".into()]).is_err(), "bad arguments are an error, not an exit");
    let _ = std::fs::remove_dir_all(&dir);
}
