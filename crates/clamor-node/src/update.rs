//! Self-update from GitHub releases.
//!
//! A release carries `manifest.json` (every build: target, variant, file, sha256, size) and
//! `manifest.json.sig`, an Ed25519 signature made in CI with the release key. This binary has
//! the public key built in, so it installs only what that key signed:
//!
//! 1. ask GitHub for the latest release; stop unless it is newer than this build;
//! 2. download the manifest and its signature; verify the signature; check the manifest's version
//!    matches the release (an old signed manifest cannot be replayed under a new tag);
//! 3. pick the build with this binary's target triple and GPU variant;
//! 4. download it; check size and SHA-256; unpack; keep the current binary as
//!    `clamor-node.previous` (see `rollback`); swap the new one in.
use crate::build_info::BuildInfo;
use anyhow::{anyhow, bail, Context, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Public half of the release signing key (Ed25519). The private half is the
/// CLAMOR_RELEASE_SIGNING_KEY secret in the GitHub repository.
pub const RELEASE_PUBKEY: [u8; 32] = [
    0xbe, 0x0a, 0x69, 0x94, 0x43, 0xc4, 0x01, 0x03, 0xa0, 0x64, 0x3c, 0x9d, 0xab, 0x2e, 0x83, 0xa3, 0x50, 0x4d, 0xad, 0x89, 0x87, 0x07, 0x4a, 0x23, 0x8c, 0x47, 0x6d, 0x27, 0x21, 0xba, 0x32, 0xe6,
];
pub const LATEST_RELEASE_API: &str = "https://api.github.com/repos/amruthwo/Clamor/releases/latest";
const MAX_DOWNLOAD: u64 = 512 << 20;

/// Where to look and which key to trust (tests substitute their own).
#[derive(Clone)]
pub struct Source {
    pub latest_release_api: String,
    pub pubkey: [u8; 32],
}
impl Source {
    pub fn github() -> Self {
        Self { latest_release_api: LATEST_RELEASE_API.into(), pubkey: RELEASE_PUBKEY }
    }
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    html_url: Option<String>,
    assets: Vec<GhAsset>,
}
#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub commit: String,
    pub assets: Vec<ManifestAsset>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestAsset {
    pub target: String,
    pub variant: String,
    pub cpu: String,
    pub file: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct Available {
    pub version: String,
    pub asset: ManifestAsset,
    pub url: String,
    pub release_page: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Check {
    UpToDate { latest: String },
    Available(Available),
    /// A newer release exists but has no build for this target and variant.
    NoMatchingBuild { latest: String },
    /// Nothing has been published yet.
    NoReleases,
}

/// `a` is a newer version than `b` (numeric major.minor.patch; anything after `-` ignored).
pub fn is_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Option<(u64, u64, u64)> {
        let core = v.trim_start_matches('v').split(['-', '+']).next()?;
        let mut it = core.split('.').map(|x| x.parse::<u64>().ok());
        Some((it.next()??, it.next().unwrap_or(Some(0))?, it.next().unwrap_or(Some(0))?))
    };
    matches!((parse(a), parse(b)), (Some(x), Some(y)) if x > y)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(300))).user_agent(concat!("clamor-node/", env!("CARGO_PKG_VERSION"))).build().into()
}

fn get(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>> {
    let mut resp = agent.get(url).header("Accept", "application/vnd.github+json, application/octet-stream").call().with_context(|| format!("GET {url}"))?;
    let mut buf = Vec::new();
    resp.body_mut().as_reader().take(MAX_DOWNLOAD + 1).read_to_end(&mut buf)?;
    if buf.len() as u64 > MAX_DOWNLOAD {
        bail!("{url}: larger than {MAX_DOWNLOAD} bytes");
    }
    Ok(buf)
}

pub fn verify_signature(pubkey: &[u8; 32], message: &[u8], sig: &[u8]) -> Result<()> {
    let key = VerifyingKey::from_bytes(pubkey).map_err(|e| anyhow!("bad release key: {e}"))?;
    let sig: [u8; 64] = sig.try_into().map_err(|_| anyhow!("signature must be 64 bytes, got {}", sig.len()))?;
    key.verify_strict(message, &Signature::from_bytes(&sig)).map_err(|_| anyhow!("manifest signature is not valid for the Clamor release key"))
}

/// Is there a newer release with a build for `me`?
pub fn check(src: &Source, me: &BuildInfo) -> Result<Check> {
    let agent = agent();
    let listing = match get(&agent, &src.latest_release_api) {
        Ok(b) => b,
        Err(e) if e.chain().any(|c| matches!(c.downcast_ref::<ureq::Error>(), Some(ureq::Error::StatusCode(404)))) => return Ok(Check::NoReleases),
        Err(e) => return Err(e),
    };
    let rel: GhRelease = serde_json::from_slice(&listing).context("parsing the release listing")?;
    let latest = rel.tag_name.trim_start_matches('v').to_string();
    if !is_newer(&latest, &me.version) {
        return Ok(Check::UpToDate { latest });
    }
    let url_of = |name: &str| rel.assets.iter().find(|a| a.name == name).map(|a| a.browser_download_url.clone());
    let manifest_bytes = get(&agent, &url_of("manifest.json").ok_or_else(|| anyhow!("release {latest} has no manifest.json"))?)?;
    let sig = get(&agent, &url_of("manifest.json.sig").ok_or_else(|| anyhow!("release {latest} has no manifest.json.sig"))?)?;
    verify_signature(&src.pubkey, &manifest_bytes, &sig)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).context("parsing manifest.json")?;
    if manifest.version != latest {
        bail!("release {latest} carries a manifest for {} (refusing a mismatched manifest)", manifest.version);
    }
    let Some(asset) = manifest.assets.iter().find(|a| a.target == me.target && a.variant == me.variant).cloned() else {
        return Ok(Check::NoMatchingBuild { latest });
    };
    let url = url_of(&asset.file).ok_or_else(|| anyhow!("manifest lists {} but the release has no such file", asset.file))?;
    Ok(Check::Available(Available { version: latest, asset, url, release_page: rel.html_url }))
}

fn backup_path(exe: &Path) -> PathBuf {
    let name = exe.file_name().and_then(|n| n.to_str()).unwrap_or("clamor-node");
    match name.strip_suffix(".exe") {
        Some(stem) => exe.with_file_name(format!("{stem}.previous.exe")),
        None => exe.with_file_name(format!("{name}.previous")),
    }
}

/// Pull `clamor-node` and `clamor-muster` out of a release archive into `dir`.
fn unpack(archive: &[u8], file_name: &str, dir: &Path) -> Result<Vec<PathBuf>> {
    let wanted = |p: &Path| p.file_name().and_then(|n| n.to_str()).map(|n| matches!(n, "clamor-node" | "clamor-node.exe" | "clamor-muster" | "clamor-muster.exe")).unwrap_or(false);
    let mut out = Vec::new();
    if file_name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
        for i in 0..z.len() {
            let mut f = z.by_index(i)?;
            let Some(p) = f.enclosed_name() else { continue };
            if f.is_file() && wanted(&p) {
                let dest = dir.join(p.file_name().unwrap());
                std::io::copy(&mut f, &mut std::fs::File::create(&dest)?)?;
                out.push(dest);
            }
        }
    } else {
        let mut t = tar::Archive::new(flate2::read::GzDecoder::new(archive));
        for e in t.entries()? {
            let mut e = e?;
            let p = e.path()?.into_owned();
            if e.header().entry_type().is_file() && wanted(&p) {
                let dest = dir.join(p.file_name().unwrap());
                e.unpack(&dest)?;
                out.push(dest);
            }
        }
    }
    if !out.iter().any(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("clamor-node"))) {
        bail!("{file_name} does not contain clamor-node");
    }
    Ok(out)
}

/// Download, verify and install `av` over `exe` (normally `std::env::current_exe()`).
/// `clamor-muster` next to it is updated too. Returns the backup path.
pub fn apply(av: &Available, exe: &Path) -> Result<PathBuf> {
    let bytes = get(&agent(), &av.url)?;
    if bytes.len() as u64 != av.asset.size {
        bail!("{}: got {} bytes, the signed manifest says {}", av.asset.file, bytes.len(), av.asset.size);
    }
    let sha = hex::encode(Sha256::digest(&bytes));
    if !sha.eq_ignore_ascii_case(&av.asset.sha256) {
        bail!("{}: SHA-256 {sha} does not match the signed manifest", av.asset.file);
    }
    let dir = exe.parent().ok_or_else(|| anyhow!("{} has no parent directory", exe.display()))?;
    // Unpack beside the binary, so the final rename stays on one filesystem.
    let tmp = dir.join(format!(".clamor-update-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)?;
    let result = (|| -> Result<PathBuf> {
        let files = unpack(&bytes, &av.asset.file, &tmp)?;
        let backup = backup_path(exe);
        std::fs::copy(exe, &backup).with_context(|| format!("backing up {}", exe.display()))?;
        for f in &files {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(f, std::fs::Permissions::from_mode(0o755))?;
            }
            let name = f.file_name().unwrap();
            let is_node = name.to_str().is_some_and(|n| n.starts_with("clamor-node"));
            if is_node {
                if std::env::current_exe().ok().and_then(|c| c.canonicalize().ok()) == exe.canonicalize().ok() {
                    // Replacing the running program (Windows cannot overwrite it in place).
                    self_replace::self_replace(f).context("replacing the running clamor-node")?;
                } else {
                    std::fs::rename(f, exe)?;
                }
            } else {
                let dest = dir.join(name);
                if dest.exists() {
                    std::fs::rename(f, &dest)?;
                }
            }
        }
        Ok(backup)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// Put back the binary that the last update replaced.
pub fn rollback(exe: &Path) -> Result<()> {
    let backup = backup_path(exe);
    if !backup.exists() {
        bail!("no previous version at {}", backup.display());
    }
    if std::env::current_exe().ok().and_then(|c| c.canonicalize().ok()) == exe.canonicalize().ok() {
        self_replace::self_replace(&backup)?;
        std::fs::remove_file(&backup)?;
    } else {
        std::fs::rename(&backup, exe)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn version_ordering() {
        assert!(is_newer("0.4.1", "0.4.0"));
        assert!(is_newer("v1.0.0", "0.9.9"));
        assert!(is_newer("0.10.0", "0.9.0"), "numeric, not lexical");
        assert!(!is_newer("0.4.0", "0.4.0"));
        assert!(!is_newer("0.3.9", "0.4.0"), "never downgrade");
        assert!(!is_newer("garbage", "0.4.0"));
        assert!(is_newer("0.5.0-rc1", "0.4.0"));
    }

    /// A manifest signed with the real release key by OpenSSL, exactly as the release workflow
    /// signs (`openssl pkeyutl -sign -rawin`), must verify against the built-in key. Catches a
    /// signature-format mismatch or a key rotated in one place but not the other.
    #[test]
    fn openssl_signatures_from_the_release_key_verify() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let body = std::fs::read(format!("{dir}/manifest-sample.json")).unwrap();
        let sig = std::fs::read(format!("{dir}/manifest-sample.json.sig")).unwrap();
        verify_signature(&RELEASE_PUBKEY, &body, &sig).unwrap();
        let mut tampered = body.clone();
        tampered[2] ^= 1;
        assert!(verify_signature(&RELEASE_PUBKEY, &tampered, &sig).is_err());
    }

    #[test]
    fn the_built_in_key_is_a_valid_ed25519_key() {
        assert!(VerifyingKey::from_bytes(&RELEASE_PUBKEY).is_ok());
    }

    /// Serve canned files over HTTP. `files` gets the server's base URL, so listings can link
    /// to their own assets.
    fn serve(files: impl FnOnce(&str) -> Vec<(String, Vec<u8>)>) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let files = files(&base);
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { continue };
                let mut req = [0u8; 4096];
                let n = s.read(&mut req).unwrap_or(0);
                let line = String::from_utf8_lossy(&req[..n]);
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                match files.iter().find(|(p, _)| *p == path) {
                    Some((_, body)) => {
                        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                        let _ = s.write_all(body);
                    }
                    None => {
                        let _ = write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    }
                }
            }
        });
        base
    }

    fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut b = tar::Builder::new(flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast()));
        for (name, data) in files {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            b.append_data(&mut h, format!("clamor-test/{name}"), *data).unwrap();
        }
        b.into_inner().unwrap().finish().unwrap()
    }

    fn me() -> BuildInfo {
        BuildInfo { version: "0.4.0".into(), target: "x86_64-unknown-linux-gnu".into(), variant: "vulkan".into(), cpu: "x86-64-v3".into(), commit: "test".into() }
    }

    fn manifest_for(version: &str, archive: &[u8]) -> Manifest {
        let m = me();
        Manifest {
            version: version.into(),
            commit: "abc".into(),
            assets: vec![
                ManifestAsset { target: "aarch64-apple-darwin".into(), variant: "metal".into(), cpu: "generic".into(), file: "other.tar.gz".into(), sha256: "00".into(), size: 1 },
                ManifestAsset { target: m.target, variant: m.variant, cpu: m.cpu, file: "build.tar.gz".into(), sha256: hex::encode(Sha256::digest(archive)), size: archive.len() as u64 },
            ],
        }
    }

    /// A fake GitHub release `tag` serving `manifest_body` with signature `sig`.
    fn release(tag: &str, manifest_body: Vec<u8>, sig: Vec<u8>, archive: Vec<u8>) -> String {
        let tag = tag.to_string();
        let base = serve(move |base| {
            let listing = serde_json::json!({
                "tag_name": tag,
                "html_url": "https://example.invalid/release",
                "assets": [
                    { "name": "manifest.json", "browser_download_url": format!("{base}/manifest.json") },
                    { "name": "manifest.json.sig", "browser_download_url": format!("{base}/manifest.json.sig") },
                    { "name": "build.tar.gz", "browser_download_url": format!("{base}/build.tar.gz") },
                ]
            });
            vec![
                ("/latest".into(), listing.to_string().into_bytes()),
                ("/manifest.json".into(), manifest_body),
                ("/manifest.json.sig".into(), sig),
                ("/build.tar.gz".into(), archive),
            ]
        });
        format!("{base}/latest")
    }

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    /// A correctly signed release of `version`.
    fn signed(version: &str, archive: &[u8], signer: &SigningKey) -> String {
        let body = serde_json::to_vec(&manifest_for(version, archive)).unwrap();
        let sig = signer.sign(&body).to_bytes().to_vec();
        release(&format!("v{version}"), body, sig, archive.to_vec())
    }

    fn source(api: String, trusted: &SigningKey) -> Source {
        Source { latest_release_api: api, pubkey: trusted.verifying_key().to_bytes() }
    }

    fn installed(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("clamor-update-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("clamor-node");
        std::fs::write(&exe, b"old node").unwrap();
        std::fs::write(dir.join("clamor-muster"), b"old muster").unwrap();
        (dir, exe)
    }

    #[test]
    fn installs_a_signed_newer_build_and_rolls_back() {
        let archive = tar_gz(&[("clamor-node", b"new node"), ("clamor-muster", b"new muster"), ("README.md", b"readme")]);
        let src = source(signed("0.9.0", &archive, &key(1)), &key(1));
        let Check::Available(av) = check(&src, &me()).unwrap() else { panic!("expected an update") };
        assert_eq!((av.version.as_str(), av.asset.variant.as_str()), ("0.9.0", "vulkan"), "picked this target and variant");

        let (dir, exe) = installed("ok");
        let backup = apply(&av, &exe).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), b"new node");
        assert_eq!(std::fs::read(dir.join("clamor-muster")).unwrap(), b"new muster");
        assert_eq!(std::fs::read(&backup).unwrap(), b"old node");
        assert!(!dir.join("README.md").exists(), "only the binaries are installed");
        rollback(&exe).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), b"old node");
    }

    #[test]
    fn refuses_anything_not_signed_by_the_trusted_key() {
        let archive = tar_gz(&[("clamor-node", b"evil")]);
        // Signed, but by another key.
        let src = source(signed("0.9.0", &archive, &key(2)), &key(1));
        assert!(check(&src, &me()).unwrap_err().to_string().contains("signature"));
        // Signed by the right key, then altered (a swapped checksum).
        let mut m = manifest_for("0.9.0", &archive);
        let sig = key(1).sign(&serde_json::to_vec(&m).unwrap()).to_bytes().to_vec();
        m.assets[1].sha256 = "11".repeat(32);
        let src = source(release("v0.9.0", serde_json::to_vec(&m).unwrap(), sig, archive.clone()), &key(1));
        assert!(check(&src, &me()).unwrap_err().to_string().contains("signature"));
        // A truncated signature.
        let body = serde_json::to_vec(&manifest_for("0.9.0", &archive)).unwrap();
        let src = source(release("v0.9.0", body, vec![1, 2, 3], archive), &key(1));
        assert!(check(&src, &me()).unwrap_err().to_string().contains("64 bytes"));
    }

    #[test]
    fn a_signed_manifest_replayed_under_another_release_is_refused() {
        // Validly signed manifest for 0.5.0, served as release v0.9.0.
        let archive = tar_gz(&[("clamor-node", b"x")]);
        let body = serde_json::to_vec(&manifest_for("0.5.0", &archive)).unwrap();
        let sig = key(1).sign(&body).to_bytes().to_vec();
        let src = source(release("v0.9.0", body, sig, archive), &key(1));
        assert!(check(&src, &me()).unwrap_err().to_string().contains("mismatched manifest"));
    }

    #[test]
    fn refuses_a_download_that_does_not_match_the_manifest() {
        let archive = tar_gz(&[("clamor-node", b"new node")]);
        let src = source(signed("0.9.0", &archive, &key(1)), &key(1));
        let Check::Available(mut av) = check(&src, &me()).unwrap() else { panic!() };
        av.asset.sha256 = "22".repeat(32); // as if the file on the server had been swapped
        let (_, exe) = installed("badsha");
        assert!(apply(&av, &exe).unwrap_err().to_string().contains("SHA-256"));
        assert_eq!(std::fs::read(&exe).unwrap(), b"old node", "nothing installed");
    }

    #[test]
    fn no_releases_yet_is_not_an_error() {
        let base = serve(|_| vec![]);
        assert!(matches!(check(&source(format!("{base}/latest"), &key(1)), &me()).unwrap(), Check::NoReleases));
    }

    #[test]
    fn same_or_older_releases_and_other_platforms_are_not_updates() {
        let archive = tar_gz(&[("clamor-node", b"x")]);
        for v in ["0.4.0", "0.3.0"] {
            let src = source(signed(v, &archive, &key(1)), &key(1));
            assert!(matches!(check(&src, &me()).unwrap(), Check::UpToDate { .. }), "{v}");
        }
        let src = source(signed("0.9.0", &archive, &key(1)), &key(1));
        let cpu_build = BuildInfo { variant: "cpu".into(), ..me() };
        assert!(matches!(check(&src, &cpu_build).unwrap(), Check::NoMatchingBuild { .. }));
    }
}

// ---------------------------------------------------------------- background checks

/// `--no-update-check` / `--auto-update`, shared by the coordinator and members.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// Look for new releases once a day (contacts api.github.com).
    pub check: bool,
    /// Install them when idle, then restart.
    pub auto: bool,
}

/// Shared "is there a newer release?" state, filled in by `spawn_checker`.
pub type UpdateFlag = std::sync::Arc<std::sync::Mutex<Option<String>>>;

/// Check GitHub shortly after start, then once a day. Records the newer version (if any) for
/// the dashboard and for `--auto-update`. Failures are logged and retried at the next check.
pub fn spawn_checker(state: UpdateFlag) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        loop {
            let me = crate::build_info::current();
            match tokio::task::spawn_blocking(move || check(&Source::github(), &me)).await {
                Ok(Ok(Check::Available(av))) => {
                    let mut s = state.lock().unwrap();
                    if s.as_deref() != Some(av.version.as_str()) {
                        println!("update available: clamor {} (run `clamor-node update`)", av.version);
                    }
                    *s = Some(av.version);
                }
                Ok(Ok(_)) => *state.lock().unwrap() = None,
                Ok(Err(e)) => eprintln!("update check failed: {e:#}"),
                Err(e) => eprintln!("update check failed: {e}"),
            }
            tokio::time::sleep(Duration::from_secs(24 * 3600)).await;
        }
    });
}

/// Install the latest release over the running binary, then restart with the same arguments.
/// Returns only on failure.
pub async fn update_and_restart() -> Result<()> {
    let me = crate::build_info::current();
    let exe = std::env::current_exe()?;
    let exe2 = exe.clone();
    let done = tokio::task::spawn_blocking(move || -> Result<Option<String>> {
        match check(&Source::github(), &me)? {
            Check::Available(av) => {
                apply(&av, &exe2)?;
                Ok(Some(av.version))
            }
            _ => Ok(None),
        }
    })
    .await??;
    let Some(v) = done else { bail!("no update to apply") };
    println!("updated to clamor {v}; restarting");
    restart(&exe)
}

/// Replace this process with a fresh run of `exe` and the same arguments.
fn restart(exe: &Path) -> Result<()> {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = std::process::Command::new(exe).args(&args).exec();
        Err(anyhow!("restarting {}: {e}", exe.display()))
    }
    #[cfg(not(unix))]
    {
        std::process::Command::new(exe).args(&args).spawn().with_context(|| format!("restarting {}", exe.display()))?;
        std::process::exit(0)
    }
}
