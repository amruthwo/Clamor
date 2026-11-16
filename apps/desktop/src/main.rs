//! Clamor desktop app: a native window and tray icon around `clamor-node`.
//!
//! The app does not contain the engine: it runs the `clamor-node` program that ships next to it
//! (as a coordinator or a member) and shows the swarm's web dashboard. First run asks whether to
//! start a swarm or join one; after that the app starts the same way every time.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, State, WindowEvent};

const LOG_LINES: usize = 400;
const PORTAL_PORT: u16 = 7421;
const CONTROL_PORT: u16 = 7420;

/// What the app runs, remembered between starts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Setup {
    /// "coordinator" or "member"; None until the first-run setup is done.
    mode: Option<String>,
    /// The swarm's name.
    swarm: Option<String>,
}

#[derive(Default)]
struct Node {
    child: Option<Child>,
    log: VecDeque<String>,
    /// The swarm's passphrase, as the coordinator printed it.
    passphrase: Option<String>,
    joined: bool,
    exited: Option<String>,
}

type Shared = Arc<Mutex<Node>>;

/// Whether the tray icon could be created.
struct HasTray(bool);

fn app_dir() -> PathBuf {
    dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("clamor")
}

fn setup_path() -> PathBuf {
    app_dir().join("desktop-app.json")
}

fn load_setup() -> Setup {
    std::fs::read(setup_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save_setup(s: &Setup) -> Result<(), String> {
    std::fs::create_dir_all(app_dir()).map_err(|e| e.to_string())?;
    std::fs::write(setup_path(), serde_json::to_vec_pretty(s).unwrap()).map_err(|e| e.to_string())
}

/// Where the app keeps model files.
fn models_dir() -> PathBuf {
    let d = dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("clamor").join("models");
    let _ = std::fs::create_dir_all(&d);
    d
}

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

/// Start `clamor-node` with `args`, collecting its output.
fn spawn_node(shared: &Shared, args: &[String]) -> Result<(), String> {
    stop_node(shared);
    let mut cmd = node_cmd()?;
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = spawn_from_supervisor(cmd).map_err(|e| format!("could not start clamor-node: {e}"))?;
    let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    {
        let mut n = shared.lock().unwrap();
        *n = Node { child: Some(child), ..Default::default() };
    }
    for stream in [Box::new(out) as Box<dyn std::io::Read + Send>, Box::new(err)] {
        let shared = shared.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let mut n = shared.lock().unwrap();
                if let Some(p) = line.split("passphrase: ").nth(1).filter(|_| line.starts_with("swarm '")) {
                    n.passphrase = Some(p.trim().to_string());
                }
                if line.starts_with("joined swarm") || line.starts_with("web portal on") {
                    n.joined = true;
                }
                n.log.push_back(line);
                while n.log.len() > LOG_LINES {
                    n.log.pop_front();
                }
            }
        });
    }
    Ok(())
}

/// Stop `clamor-node`, letting a member leave the swarm gracefully (it finishes its work).
fn stop_node(shared: &Shared) {
    let child = shared.lock().unwrap().child.take();
    let Some(mut child) = child else { return };
    #[cfg(unix)]
    unsafe {
        libc::kill(child.id() as i32, libc::SIGINT);
    }
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(5) {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Whether the node is still running; records why it stopped if it did.
fn check_alive(n: &mut Node) -> bool {
    let Some(child) = n.child.as_mut() else { return false };
    match child.try_wait() {
        Ok(None) => true,
        Ok(Some(st)) => {
            let last = n.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
            n.exited = Some(format!("clamor-node stopped ({st}): {last}"));
            n.child = None;
            false
        }
        Err(e) => {
            n.exited = Some(e.to_string());
            n.child = None;
            false
        }
    }
}

fn base_args() -> Vec<String> {
    vec!["--models".into(), models_dir().display().to_string()]
}

fn start_from_setup(shared: &Shared, s: &Setup) -> Result<(), String> {
    let mut args = Vec::new();
    match s.mode.as_deref() {
        Some("coordinator") => {
            args.extend(["coordinator".into(), "--swarm".into(), s.swarm.clone().unwrap_or_else(|| "home".into())]);
        }
        Some("member") => {
            args.push("join".into());
            if let Some(name) = &s.swarm {
                args.extend(["--swarm".into(), name.clone()]);
            }
        }
        _ => return Err("not set up yet".into()),
    }
    args.extend(base_args());
    spawn_node(shared, &args)
}

/// Run `clamor-node swarms --json`.
fn swarms_json(scan: bool) -> Result<Value, String> {
    let mut c = node_cmd()?;
    c.args(["swarms", "--json"]);
    if scan {
        c.arg("--scan");
    }
    let out = c.output().map_err(|e| e.to_string())?;
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{e}: {}", String::from_utf8_lossy(&out.stderr)))
}

/// The dashboard to show: our own on the coordinator; the coordinator's on a member.
fn portal_url(s: &Setup) -> Option<String> {
    match s.mode.as_deref() {
        Some("coordinator") => Some(format!("http://127.0.0.1:{PORTAL_PORT}/")),
        Some("member") => {
            let v = swarms_json(false).ok()?;
            let joined = v["joined"].as_array()?;
            let p = joined.iter().find(|p| s.swarm.as_deref().is_none_or(|w| p["name"] == w))?;
            let addr = p["address"].as_str()?;
            let host = addr.rsplit_once(':').map(|(h, _)| h).unwrap_or(addr);
            Some(format!("http://{host}:{PORTAL_PORT}/"))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------- commands for the UI

#[tauri::command]
fn status(shared: State<Shared>) -> Value {
    let setup = load_setup();
    let mut n = shared.lock().unwrap();
    let running = check_alive(&mut n);
    json!({
        "node_found": node_path().is_some(),
        "setup": setup,
        "running": running,
        "joined": n.joined,
        "exited": n.exited,
        "passphrase": n.passphrase,
        "log": n.log.iter().rev().take(60).rev().collect::<Vec<_>>(),
        "models_dir": models_dir(),
    })
}

#[tauri::command]
async fn swarms(scan: bool) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || swarms_json(scan)).await.map_err(|e| e.to_string())?
}

#[tauri::command]
fn dashboard_url() -> Option<String> {
    portal_url(&load_setup())
}

/// Wait until the node says it is in (or stops), for up to `secs`.
fn wait_joined(shared: &Shared, secs: u64) -> Result<(), String> {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        {
            let mut n = shared.lock().unwrap();
            if n.joined {
                return Ok(());
            }
            if !check_alive(&mut n) {
                return Err(n.exited.clone().unwrap_or_else(|| "clamor-node stopped".into()));
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err("no answer from the swarm yet; see the log".into())
}

#[tauri::command]
async fn start_swarm(shared: State<'_, Shared>, name: String, passphrase: Option<String>) -> Result<(), String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("give the swarm a name".into());
    }
    let mut args = vec!["coordinator".to_string(), "--swarm".into(), name.clone()];
    if let Some(p) = passphrase.filter(|p| !p.trim().is_empty()) {
        args.extend(["--passphrase".into(), p]);
    }
    args.extend(base_args());
    let shared = shared.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        spawn_node(&shared, &args)?;
        wait_joined(&shared, 60)?;
        save_setup(&Setup { mode: Some("coordinator".into()), swarm: Some(name) })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn join_swarm(shared: State<'_, Shared>, swarm: Option<String>, address: Option<String>, secret: String) -> Result<(), String> {
    let secret = secret.trim().to_string();
    let mut args = vec!["join".to_string()];
    if let Some(a) = address.filter(|a| !a.trim().is_empty()) {
        let a = a.trim();
        args.push(if a.contains(':') { a.to_string() } else { format!("{a}:{CONTROL_PORT}") });
    } else if let Some(s) = &swarm {
        args.extend(["--swarm".into(), s.clone()]);
    }
    // Six digits are a one-time code; anything else is the passphrase.
    let digits: String = secret.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() == 6 && secret.chars().all(|c| c.is_ascii_digit() || c == '-' || c == ' ') {
        args.extend(["--code".into(), format!("{}-{}", &digits[..3], &digits[3..])]);
    } else if !secret.is_empty() {
        args.extend(["--passphrase".into(), secret]);
    }
    args.extend(base_args());
    let shared = shared.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        spawn_node(&shared, &args)?;
        wait_joined(&shared, 45)?;
        // The swarm's name, from what the node now remembers.
        let name = swarms_json(false).ok().and_then(|v| v["joined"][0]["name"].as_str().map(str::to_string)).or(swarm);
        save_setup(&Setup { mode: Some("member".into()), swarm: name })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn restart(shared: State<'_, Shared>) -> Result<(), String> {
    let shared = shared.inner().clone();
    tauri::async_runtime::spawn_blocking(move || start_from_setup(&shared, &load_setup())).await.map_err(|e| e.to_string())?
}

/// Stop taking part and forget the setup (the swarm itself is still remembered by clamor-node,
/// so joining it again needs no passphrase).
#[tauri::command]
async fn reset(shared: State<'_, Shared>) -> Result<(), String> {
    let shared = shared.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        stop_node(&shared);
        save_setup(&Setup::default())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn autostart_enabled(app: AppHandle) -> bool {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().unwrap_or(false)
}

#[tauri::command]
fn set_autostart(app: AppHandle, on: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let a = app.autolaunch();
    if on { a.enable() } else { a.disable() }.map_err(|e| e.to_string())
}

fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn main() {
    let shared: Shared = Default::default();
    let for_exit = shared.clone();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .plugin(tauri_plugin_autostart::init(tauri_plugin_autostart::MacosLauncher::LaunchAgent, Some(vec!["--background"])))
        .plugin(tauri_plugin_opener::init())
        .manage(shared.clone())
        .invoke_handler(tauri::generate_handler![status, swarms, dashboard_url, start_swarm, join_swarm, restart, reset, autostart_enabled, set_autostart])
        .setup(move |app| {
            // Tray: open the window, open the models folder, quit.
            let open = MenuItem::with_id(app, "open", "Open Clamor", true, None::<&str>)?;
            let models = MenuItem::with_id(app, "models", "Open models folder", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Clamor", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &models, &quit])?;
            // Some desktops (and sandboxes without the tray library) have no tray: carry on
            // without it, with the window as the only way in.
            let tray = TrayIconBuilder::with_id("clamor")
                .icon(app.default_window_icon().cloned().expect("app icon"))
                .tooltip("Clamor")
                .menu(&menu)
                .on_menu_event(|app, ev| match ev.id().as_ref() {
                    "open" => show_main(app),
                    "models" => {
                        use tauri_plugin_opener::OpenerExt;
                        let _ = app.opener().open_path(models_dir().display().to_string(), None::<&str>);
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app);
            let has_tray = tray.is_ok();
            if let Err(e) = tray {
                eprintln!("no tray icon: {e}");
            }
            app.manage(HasTray(has_tray));
            // Started at login: stay in the tray.
            if has_tray && std::env::args().any(|a| a == "--background") {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
            // Set up before: carry on as last time.
            let s = load_setup();
            if s.mode.is_some() {
                if let Err(e) = start_from_setup(&shared, &s) {
                    shared.lock().unwrap().exited = Some(e);
                }
            }
            Ok(())
        })
        .on_window_event(|w, ev| {
            // Closing the window keeps Clamor running in the tray (without a tray, it quits).
            if let WindowEvent::CloseRequested { api, .. } = ev {
                if w.app_handle().try_state::<HasTray>().is_some_and(|t| t.0) {
                    api.prevent_close();
                    let _ = w.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("could not start the app");
    app.run(move |_, ev| {
        if let tauri::RunEvent::Exit = ev {
            stop_node(&for_exit);
        }
    });
}
