//! Updating the app itself: find out how it is installed, ask the signed release manifest for
//! a newer installer of that kind, download it (size and SHA-256 checked against the manifest,
//! whose signature is checked against the Clamor release key), and install it the way that kind
//! is installed. Nothing happens without the person pressing "Update now".
//!
//! | installed as | installs by                                                    |
//! |--------------|----------------------------------------------------------------|
//! | nsis (Win)   | running the new installer (passive), then the app quits         |
//! | dmg (macOS)  | swapping the .app bundle from the mounted image, then restarting|
//! | appimage     | replacing the AppImage file, then restarting                    |
//! | rpm / deb    | `pkexec dnf/zypper/apt install` (asks for the password)         |
//! | flatpak      | opening the new bundle in the software app (Discover, Software) |
//! | apk          | handing the APK to Android's installer ("Update this app?")     |
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

/// How this app was installed, or None (a development build: nothing to update).
pub fn install_kind() -> Option<&'static str> {
    if cfg!(target_os = "android") {
        return Some("apk");
    }
    if cfg!(target_os = "windows") {
        return Some("nsis");
    }
    if cfg!(target_os = "macos") {
        return bundle_path().map(|_| "dmg");
    }
    if std::env::var_os("FLATPAK_ID").is_some() {
        return Some("flatpak");
    }
    if std::env::var_os("APPIMAGE").is_some() {
        return Some("appimage");
    }
    let exe = std::env::current_exe().ok()?;
    let owned = |cmd: &str, arg: &str| Command::new(cmd).args([arg, &exe.display().to_string()]).output().is_ok_and(|o| o.status.success());
    if owned("rpm", "-qf") {
        Some("rpm")
    } else if owned("dpkg", "-S") {
        Some("deb")
    } else {
        None
    }
}

/// The running .app bundle on macOS.
fn bundle_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find(|p| p.extension().is_some_and(|e| e == "app")).map(Path::to_path_buf)
}

/// `{"update": null}` or `{"update": {version, notes, asset, ...}}`.
pub fn check(current: &str) -> Result<Value, String> {
    let Some(kind) = install_kind() else { return Ok(serde_json::json!({ "update": null, "kind": null })) };
    let mut v = check_kind(kind, current)?;
    v["kind"] = kind.into();
    Ok(v)
}

#[cfg(not(target_os = "android"))]
fn check_kind(kind: &str, current: &str) -> Result<Value, String> {
    crate::runner::app_update(&["check", "--kind", kind, "--current", current])
}

#[cfg(target_os = "android")]
fn check_kind(kind: &str, current: &str) -> Result<Value, String> {
    let u = clamor_node::app_update_check(kind, current).map_err(|e| format!("{e:#}"))?;
    Ok(serde_json::json!({ "update": u }))
}

/// Download the update into `dir`; the verified file.
#[cfg(not(target_os = "android"))]
fn download(kind: &str, current: &str, dir: &Path) -> Result<PathBuf, String> {
    let v = crate::runner::app_update(&["download", "--kind", kind, "--current", current, "--dir", &dir.display().to_string()])?;
    v["file"].as_str().map(PathBuf::from).ok_or_else(|| "there is no newer version".to_string())
}

#[cfg(target_os = "android")]
fn download(kind: &str, current: &str, dir: &Path) -> Result<PathBuf, String> {
    let u = clamor_node::app_update_check(kind, current).map_err(|e| format!("{e:#}"))?.ok_or("there is no newer version")?;
    clamor_node::app_update_download(&u, dir).map_err(|e| format!("{e:#}"))
}

/// What the app does after installing.
pub enum After {
    /// Restart (the new version is in place).
    Restart,
    /// Quit: the installer that was started finishes the job (Windows).
    Quit,
    /// Another app takes over: open `file` with it (Flatpak's software app, Android's
    /// installer); `message` says what to do there.
    Open { file: PathBuf, message: String },
}

/// Download and install the update.
pub fn install(current: &str, dir: &Path) -> Result<After, String> {
    let kind = install_kind().ok_or("this copy of Clamor was not installed from a release, so it cannot update itself")?;
    let file = download(kind, current, dir)?;
    match kind {
        "rpm" | "deb" => {
            let (pm, args): (&str, &[&str]) = if kind == "deb" {
                ("apt-get", &["install", "-y"])
            } else if Path::new("/usr/bin/dnf").exists() || Path::new("/usr/bin/dnf5").exists() {
                ("dnf", &["install", "-y"])
            } else {
                ("zypper", &["--non-interactive", "install", "--allow-unsigned-rpm"])
            };
            // pkexec asks for the password (graphically) and runs the package manager.
            let st = Command::new("pkexec").arg(pm).args(args).arg(&file).status().map_err(|e| format!("could not run pkexec: {e}"))?;
            if !st.success() {
                return Err(format!("{pm} did not install the update ({st}); was the password dialog cancelled?"));
            }
            Ok(After::Restart)
        }
        "appimage" => {
            let target = PathBuf::from(std::env::var_os("APPIMAGE").ok_or("not running as an AppImage")?);
            let tmp = target.with_extension("new");
            std::fs::copy(&file, &tmp).map_err(|e| format!("could not write next to {}: {e}", target.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
            }
            std::fs::rename(&tmp, &target).map_err(|e| e.to_string())?;
            Ok(After::Restart)
        }
        "dmg" => {
            let bundle = bundle_path().ok_or("cannot find the running app")?;
            let mount = dir.join("mount");
            std::fs::create_dir_all(&mount).map_err(|e| e.to_string())?;
            let st = Command::new("hdiutil").args(["attach", "-nobrowse", "-readonly", "-mountpoint"]).arg(&mount).arg(&file).status().map_err(|e| e.to_string())?;
            if !st.success() {
                return Err("could not open the disk image".into());
            }
            let swap = (|| -> Result<(), String> {
                let new_app = std::fs::read_dir(&mount).map_err(|e| e.to_string())?.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "app")).ok_or("no app in the disk image")?;
                let staged = bundle.with_extension("app.new");
                let old = bundle.with_extension("app.old");
                let _ = std::fs::remove_dir_all(&staged);
                let _ = std::fs::remove_dir_all(&old);
                let st = Command::new("ditto").arg(&new_app).arg(&staged).status().map_err(|e| e.to_string())?;
                if !st.success() {
                    return Err(format!("could not copy the new version next to {}", bundle.display()));
                }
                std::fs::rename(&bundle, &old).map_err(|e| e.to_string())?;
                std::fs::rename(&staged, &bundle).map_err(|e| e.to_string())?;
                let _ = std::fs::remove_dir_all(&old);
                Ok(())
            })();
            let _ = Command::new("hdiutil").args(["detach", "-quiet"]).arg(&mount).status();
            swap.map(|_| After::Restart)
        }
        "nsis" => {
            // Passive: a progress window, no questions; it replaces the files once the app has quit.
            Command::new(&file).arg("/P").spawn().map_err(|e| format!("could not start the installer: {e}"))?;
            Ok(After::Quit)
        }
        "flatpak" => Ok(After::Open { file, message: "The new version opened in your software app: install it there, then start Clamor again.".into() }),
        "apk" => Ok(After::Open { file, message: "Android asks whether to update Clamor: tap Update.".into() }),
        other => Err(format!("cannot update a {other} installation")),
    }
}

/// Start the (new) app again after a moment, so this one can quit first (one instance at a time).
pub fn relaunch() {
    #[cfg(target_os = "macos")]
    if let Some(b) = bundle_path() {
        let _ = Command::new("sh").arg("-c").arg(format!("sleep 2; open -n {:?}", b)).spawn();
        return;
    }
    #[cfg(unix)]
    {
        let exe = std::env::var_os("APPIMAGE").map(PathBuf::from).or_else(|| std::env::current_exe().ok());
        if let Some(exe) = exe {
            // After a package upgrade the old executable is gone ("(deleted)"): start the path.
            let path = exe.display().to_string().trim_end_matches(" (deleted)").to_string();
            let _ = Command::new("sh").arg("-c").arg(format!("sleep 2; exec {path:?}")).spawn();
        }
    }
}
