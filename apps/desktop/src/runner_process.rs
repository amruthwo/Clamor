//! Desktop: clamor-node is a separate program next to the app. It is started from one
//! long-lived thread (so that on Linux it exits if the app dies), and stopped with SIGINT so a
//! member leaves the swarm gracefully.
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Runner(Child);

/// `clamor-node`: next to this program (how the installers ship it), else on the PATH.
fn node_path() -> Option<PathBuf> {
    let name = if cfg!(windows) { "clamor-node.exe" } else { "clamor-node" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(name)).find(|p| p.is_file()))
}

fn node_cmd() -> Result<Command, String> {
    let p = node_path().ok_or("clamor-node was not found next to the app or on the PATH")?;
    #[allow(unused_mut)] // changed on Windows
    let mut c = Command::new(p);
    // The app updates clamor-node together with itself.
    c.env("CLAMOR_APP_MANAGED", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    // If the app dies without quitting (killed, crashed), take clamor-node with it.
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt;
        c.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    Ok(c)
}

/// Start a process from one thread that lives as long as the app. On Linux the child is told to
/// exit when its parent dies (PR_SET_PDEATHSIG), and "parent" means the *thread* that started
/// it: starting it from a short-lived worker thread would kill it when that thread retires.
fn spawn_from_supervisor(cmd: Command) -> std::io::Result<Child> {
    type Req = (Command, std::sync::mpsc::Sender<std::io::Result<Child>>);
    static SUPERVISOR: std::sync::OnceLock<Mutex<std::sync::mpsc::Sender<Req>>> = std::sync::OnceLock::new();
    let tx = SUPERVISOR.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Req>();
        std::thread::Builder::new()
            .name("clamor-supervisor".into())
            .spawn(move || {
                for (mut cmd, reply) in rx {
                    let _ = reply.send(cmd.spawn());
                }
            })
            .expect("supervisor thread");
        Mutex::new(tx)
    });
    let (rtx, rrx) = std::sync::mpsc::channel();
    tx.lock().unwrap().send((cmd, rtx)).map_err(|_| std::io::Error::other("supervisor stopped"))?;
    rrx.recv().map_err(|_| std::io::Error::other("supervisor stopped"))?
}

pub fn node_found() -> bool {
    node_path().is_some()
}

/// Start clamor-node with `args`; every line it prints goes to `on_line`.
pub fn spawn(args: &[String], on_line: Arc<dyn Fn(String) + Send + Sync>) -> Result<Runner, String> {
    let mut cmd = node_cmd()?;
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = spawn_from_supervisor(cmd).map_err(|e| format!("could not start clamor-node: {e}"))?;
    let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    for stream in [Box::new(out) as Box<dyn std::io::Read + Send>, Box::new(err)] {
        let on_line = on_line.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                on_line(line);
            }
        });
    }
    Ok(Runner(child))
}

impl Runner {
    /// Stop it, letting a member leave gracefully (5 s, then killed).
    pub fn stop(mut self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(self.0.id() as i32, libc::SIGINT);
        }
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(5) {
            if let Ok(Some(_)) = self.0.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }

    /// Why it stopped, once it has.
    pub fn exited(&mut self) -> Option<String> {
        match self.0.try_wait() {
            Ok(None) => None,
            Ok(Some(st)) => Some(st.to_string()),
            Err(e) => Some(e.to_string()),
        }
    }
}

/// `clamor-node swarms --json`.
pub fn swarms_json(scan: bool, _config_dir: &Path) -> Result<Value, String> {
    let mut c = node_cmd()?;
    c.args(["swarms", "--json"]);
    if scan {
        c.arg("--scan");
    }
    let out = c.output().map_err(|e| e.to_string())?;
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{e}: {}", String::from_utf8_lossy(&out.stderr)))
}

/// `clamor-node app-update ARGS...` (check / download), its JSON.
pub fn app_update(args: &[&str]) -> Result<Value, String> {
    let mut c = node_cmd()?;
    c.arg("app-update").args(args);
    let out = c.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().trim_start_matches("Error: ").to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{e}: {}", String::from_utf8_lossy(&out.stderr)))
}
