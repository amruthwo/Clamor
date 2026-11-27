//! Android: clamor-node runs inside the app, on its own thread (Android does not let an app
//! start programs of its own). What it prints goes to the app's standard output, which is
//! redirected into a pipe once, so the app sees the same lines as on desktops. `shutdown()`
//! stops it the way Ctrl-C does: a member leaves the swarm gracefully.
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Listener = Arc<dyn Fn(String) + Send + Sync>;

/// Who gets the printed lines now (the running node's listener).
static LISTENER: Mutex<Option<Listener>> = Mutex::new(None);
static CAPTURE: Once = Once::new();

/// Redirect this process's stdout and stderr into a pipe read by a thread.
fn capture_output() {
    CAPTURE.call_once(|| unsafe {
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        libc::dup2(fds[1], 1);
        libc::dup2(fds[1], 2);
        let read = {
            use std::os::fd::FromRawFd;
            std::fs::File::from_raw_fd(fds[0])
        };
        std::thread::Builder::new()
            .name("clamor-output".into())
            .spawn(move || {
                for line in BufReader::new(read).lines().map_while(Result::ok) {
                    logcat(&line);
                    let l = LISTENER.lock().unwrap().clone();
                    if let Some(l) = l {
                        l(line);
                    }
                }
            })
            .ok();
    });
}

/// Also to Android's system log (tag "clamor"), readable with `adb logcat -s clamor`.
fn logcat(line: &str) {
    #[link(name = "log")]
    extern "C" {
        fn __android_log_write(prio: i32, tag: *const std::ffi::c_char, text: *const std::ffi::c_char) -> i32;
    }
    const INFO: i32 = 4;
    if let Ok(text) = std::ffi::CString::new(line) {
        unsafe {
            __android_log_write(INFO, c"clamor".as_ptr(), text.as_ptr());
        }
    }
}

pub struct Runner(Option<JoinHandle<anyhow::Result<()>>>);

pub fn node_found() -> bool {
    true
}

/// Start clamor-node in this process with `args`; every line it prints goes to `on_line`.
pub fn spawn(args: &[String], on_line: Arc<dyn Fn(String) + Send + Sync>) -> Result<Runner, String> {
    capture_output();
    *LISTENER.lock().unwrap() = Some(on_line);
    let args = args.to_vec();
    let h = std::thread::Builder::new()
        .name("clamor-node".into())
        .spawn(move || clamor_node::run_args(args))
        .map_err(|e| format!("could not start clamor-node: {e}"))?;
    Ok(Runner(Some(h)))
}

impl Runner {
    /// Stop it (a member leaves gracefully first), waiting up to 15 s.
    pub fn stop(mut self) {
        clamor_node::shutdown();
        let t0 = Instant::now();
        while self.0.as_ref().is_some_and(|h| !h.is_finished()) && t0.elapsed() < Duration::from_secs(15) {
            std::thread::sleep(Duration::from_millis(100));
        }
        if let Some(h) = self.0.take().filter(|h| h.is_finished()) {
            let _ = h.join();
        }
    }

    /// Why it stopped, once it has.
    pub fn exited(&mut self) -> Option<String> {
        if !self.0.as_ref()?.is_finished() {
            return None;
        }
        Some(match self.0.take()?.join() {
            Ok(Ok(())) => "stopped".into(),
            Ok(Err(e)) => format!("{e:#}"),
            Err(_) => "crashed".into(),
        })
    }
}

pub fn swarms_json(scan: bool, config_dir: &Path) -> Result<Value, String> {
    clamor_node::swarms_json(config_dir, scan).map_err(|e| format!("{e:#}"))
}

/// Run a one-off clamor-node command (`leave`, ...) in this process (the node is stopped).
pub fn run_once(args: &[String]) -> Result<String, String> {
    clamor_node::run_args(args.to_vec()).map(|_| String::new()).map_err(|e| format!("{e:#}"))
}
