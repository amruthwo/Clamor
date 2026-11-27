//! Clamor desktop app: a native window and tray icon around `clamor-node`.
//!
//! The app does not contain the engine: it runs the `clamor-node` program that ships next to it
//! (as a coordinator or a member) and shows the swarm's web dashboard. First run asks whether to
//! start a swarm or join one; after that the app starts the same way every time.
//!
//! On desktops `clamor-node` is a separate program next to the app (runner_process.rs); on
//! Android the app runs it in-process (runner_embedded.rs): Android does not let an app start
//! programs of its own the way desktops do.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
#[cfg(desktop)]
use tauri::menu::{Menu, MenuItem};
#[cfg(desktop)]
use tauri::tray::TrayIconBuilder;
#[cfg(desktop)]
use tauri::WindowEvent;
use tauri::{AppHandle, Manager, State};

#[cfg(not(target_os = "android"))]
#[path = "runner_process.rs"]
mod runner;
mod updater;
#[cfg(target_os = "android")]
#[path = "runner_embedded.rs"]
mod runner;

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
    runner: Option<runner::Runner>,
    log: VecDeque<String>,
    /// The swarm's passphrase, as the coordinator printed it.
    passphrase: Option<String>,
    joined: bool,
    exited: Option<String>,
    /// While asking to join: the code an admin compares.
    check_code: Option<String>,
}

type Shared = Arc<Mutex<Node>>;

/// Whether the tray icon could be created.
#[cfg(desktop)]
struct HasTray(bool);

/// Where the app keeps its things on Android (its private storage), set at start.
static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

fn app_dir() -> PathBuf {
    match DATA_DIR.get() {
        Some(d) => d.join("config"),
        None => dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("clamor"),
    }
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
    let d = match DATA_DIR.get() {
        Some(d) => d.join("models"),
        None => dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("clamor").join("models"),
    };
    let _ = std::fs::create_dir_all(&d);
    d
}



/// What clamor-node printed: kept for the Log view, and watched for the passphrase, the check
/// code and "joined".
fn on_line(shared: &Shared, line: String) {
    let mut n = shared.lock().unwrap();
    if let Some(p) = line.split("passphrase: ").nth(1).filter(|_| line.starts_with("swarm '")) {
        n.passphrase = Some(p.trim().to_string());
    }
    if let Some(c) = line.split("Check code: ").nth(1) {
        n.check_code = Some(c.trim().to_string());
    }
    if line.starts_with("joined swarm") || line.starts_with("web portal on") {
        n.joined = true;
    }
    n.log.push_back(line);
    while n.log.len() > LOG_LINES {
        n.log.pop_front();
    }
}

/// Start clamor-node with `args`.
fn spawn_node(shared: &Shared, args: &[String]) -> Result<(), String> {
    stop_node(shared);
    {
        let mut n = shared.lock().unwrap();
        *n = Node::default();
    }
    let s2 = shared.clone();
    let r = runner::spawn(args, Arc::new(move |line| on_line(&s2, line)))?;
    shared.lock().unwrap().runner = Some(r);
    Ok(())
}

/// Stop clamor-node, letting a member leave the swarm gracefully (it finishes its work).
fn stop_node(shared: &Shared) {
    let r = shared.lock().unwrap().runner.take();
    if let Some(r) = r {
        r.stop();
    }
}

/// Whether the node is still running; records why it stopped if it did.
fn check_alive(n: &mut Node) -> bool {
    let Some(r) = n.runner.as_mut() else { return false };
    match r.exited() {
        None => true,
        Some(why) => {
            let last = n.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
            n.exited = Some(format!("clamor-node stopped ({why}): {last}"));
            n.runner = None;
            false
        }
    }
}

fn base_args() -> Vec<String> {
    let mut a = vec!["--models".into(), models_dir().display().to_string()];
    // Android: everything in the app's private storage.
    if let Some(d) = DATA_DIR.get() {
        a.extend(["--config-dir".into(), app_dir().display().to_string(), "--cache-dir".into(), d.join("cache").display().to_string()]);
    }
    a
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

/// The swarms clamor-node remembers (and, with `scan`, those on the network).
fn swarms_json(scan: bool) -> Result<Value, String> {
    runner::swarms_json(scan, &app_dir())
}

/// The dashboard to show: on the coordinator its own; on a member the swarm's dashboard as
/// clamor-node serves it on this device (signed in as this device's person, encrypted to the
/// coordinator). Both are on this machine.
fn portal_url(s: &Setup) -> Option<String> {
    s.mode.as_ref().map(|_| format!("http://127.0.0.1:{PORTAL_PORT}/"))
}

// ---------------------------------------------------------------- commands for the UI

#[tauri::command]
fn status(shared: State<Shared>) -> Value {
    let setup = load_setup();
    let mut n = shared.lock().unwrap();
    let running = check_alive(&mut n);
    json!({
        "node_found": runner::node_found(),
        "setup": setup,
        "running": running,
        "joined": n.joined,
        "exited": n.exited,
        "passphrase": n.passphrase,
        "check_code": n.check_code,
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
async fn join_swarm(shared: State<'_, Shared>, swarm: Option<String>, address: Option<String>, secret: String, name: Option<String>) -> Result<(), String> {
    let secret = secret.trim().to_string();
    let mut args = vec!["join".to_string()];
    if let Some(n) = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
        args.extend(["--name".into(), n]);
    }
    if let Some(a) = address.filter(|a| !a.trim().is_empty()) {
        let a = a.trim();
        args.push(if a.contains(':') { a.to_string() } else { format!("{a}:{CONTROL_PORT}") });
    } else if let Some(s) = &swarm {
        args.extend(["--swarm".into(), s.clone()]);
    }
    // Nothing: ask to join (an admin approves it). Six digits: a one-time code. Else the passphrase.
    let asking = secret.is_empty();
    if asking {
        args.push("--ask".into());
    }
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
        wait_joined(&shared, if asking { 640 } else { 45 })?;
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

/// Is there a newer version of this app (for how it is installed here)?
#[tauri::command]
async fn update_check() -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(|| updater::check(env!("CARGO_PKG_VERSION"))).await.map_err(|e| e.to_string())?
}

/// Download, verify and install the newer version (only ever on the person's request).
/// `{"then": "restart" | "quit" | "open", "file", "message"}`: the window does the rest.
#[tauri::command]
async fn update_install(app: AppHandle, shared: State<'_, Shared>) -> Result<Value, String> {
    let dir = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("updates");
    let after = tauri::async_runtime::spawn_blocking(move || updater::install(env!("CARGO_PKG_VERSION"), &dir)).await.map_err(|e| e.to_string())??;
    Ok(match after {
        updater::After::Restart | updater::After::Quit => {
            let restart = matches!(after, updater::After::Restart);
            // Leave the swarm gracefully before the new version takes over.
            stop_node(&shared);
            if restart {
                updater::relaunch();
            }
            let a = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(300));
                a.exit(0);
            });
            serde_json::json!({ "then": if restart { "restart" } else { "quit" } })
        }
        updater::After::Open { file, message } => {
            #[cfg(desktop)]
            {
                use tauri_plugin_opener::OpenerExt;
                app.opener().open_path(file.display().to_string(), None::<&str>).map_err(|e| format!("could not open {}: {e}", file.display()))?;
            }
            // On Android the window hands the file to the installer (MainActivity's bridge).
            serde_json::json!({ "then": "open", "file": file, "message": message })
        }
    })
}

#[tauri::command]
fn autostart_enabled(app: AppHandle) -> bool {
    #[cfg(desktop)]
    {
        use tauri_plugin_autostart::ManagerExt;
        app.autolaunch().is_enabled().unwrap_or(false)
    }
    #[cfg(mobile)]
    {
        let _ = app;
        false
    }
}

#[tauri::command]
fn set_autostart(app: AppHandle, on: bool) -> Result<(), String> {
    #[cfg(desktop)]
    {
        use tauri_plugin_autostart::ManagerExt;
        let a = app.autolaunch();
        if on { a.enable() } else { a.disable() }.map_err(|e| e.to_string())
    }
    #[cfg(mobile)]
    {
        let _ = (app, on);
        Err("starting at login is set by Android".into())
    }
}

#[cfg(desktop)]
/// Linux tray icons need libayatana-appindicator (or the older libappindicator), loaded at run
/// time; without it Tauri panics instead of returning an error, so look first. Elsewhere the
/// tray is part of the system.
fn tray_library_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        ["libayatana-appindicator3.so.1", "libappindicator3.so.1", "libayatana-appindicator3.so", "libappindicator3.so"].iter().any(|name| {
            let c = std::ffi::CString::new(*name).unwrap();
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY) };
            !h.is_null()
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

#[cfg(desktop)]
fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shared: Shared = Default::default();
    let for_exit = shared.clone();
    let builder = tauri::Builder::default();
    #[cfg(desktop)]
    let builder = builder
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .plugin(tauri_plugin_autostart::init(tauri_plugin_autostart::MacosLauncher::LaunchAgent, Some(vec!["--background"])));
    #[cfg(target_os = "android")]
    let builder = builder.plugin(tauri_plugin_barcode_scanner::init());
    let builder = builder
        .plugin(tauri_plugin_opener::init())
        .manage(shared.clone())
        .invoke_handler(tauri::generate_handler![status, swarms, dashboard_url, start_swarm, join_swarm, restart, reset, autostart_enabled, set_autostart, update_check, update_install])
        .setup(move |app| {
            #[cfg(mobile)]
            {
                // Android: settings, keys, models and the cache live in the app's own storage.
                let dir = app.path().app_data_dir()?;
                std::fs::create_dir_all(&dir)?;
                // The app's service keeps the battery state here (apps cannot read it the Linux
                // way on Android); clamor-node pauses a phone on battery unless told otherwise.
                std::env::set_var("CLAMOR_POWER_FILE", dir.join("power.txt"));
                // The app updates clamor-node (built into it) together with itself.
                std::env::set_var("CLAMOR_APP_MANAGED", "1");
                let _ = DATA_DIR.set(dir);
            }
            #[cfg(desktop)]
            desktop_setup(app)?;
            // Set up before: carry on as last time.
            let s = load_setup();
            if s.mode.is_some() {
                if let Err(e) = start_from_setup(&shared, &s) {
                    shared.lock().unwrap().exited = Some(e);
                }
            }
            Ok(())
        });
    #[cfg(desktop)]
    let builder = builder.on_window_event(|w, ev| {
        // Closing the window keeps Clamor running in the tray (without a tray, it quits).
        if let WindowEvent::CloseRequested { api, .. } = ev {
            if w.app_handle().try_state::<HasTray>().is_some_and(|t| t.0) {
                api.prevent_close();
                let _ = w.hide();
            }
        }
    });
    let app = builder.build(tauri::generate_context!()).expect("could not start the app");
    app.run(move |_, ev| {
        if let tauri::RunEvent::Exit = ev {
            stop_node(&for_exit);
        }
    });
}

/// The tray (open, models folder, quit), and starting hidden when started at login.
#[cfg(desktop)]
fn desktop_setup(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let open = MenuItem::with_id(app, "open", "Open Clamor", true, None::<&str>)?;
    let models = MenuItem::with_id(app, "models", "Open models folder", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Clamor", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &models, &quit])?;
    // Some desktops (and sandboxes without the tray library) have no tray: carry on without
    // it, with the window as the only way in.
    let tray = if !tray_library_available() {
        Err(tauri::Error::from(std::io::Error::other("no tray library (libayatana-appindicator3) on this system")))
    } else {
        TrayIconBuilder::with_id("clamor")
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
            .build(app)
    };
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
    Ok(())
}
