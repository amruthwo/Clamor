# Updating Clamor

## The app

The Clamor app (desktop and Android) checks for a new version shortly after it starts and then
once a day. When there is one, a strip at the top of the window says so:
**Clamor 0.7.0 is available.** Press **Update…** to read what is new, then **Update now**.
Nothing is downloaded or installed until you press it; **Later** hides the strip until the
next version.

The app picks the file that matches how it is installed on that device, so you never choose a
download yourself:

| Installed from | What happens after **Update now** |
|---|---|
| Windows setup (`.exe`) | The new installer runs with a progress window; Clamor closes while it installs and starts again afterwards. |
| macOS disk image (`.dmg`) | The app is replaced and restarts. |
| AppImage | The AppImage file is replaced and Clamor restarts. |
| `.rpm` (Fedora, openSUSE) | The new package is installed with `dnf` or `zypper`; your computer asks for your password. Then Clamor restarts. |
| `.deb` (Debian, Ubuntu) | The same, with `apt`. |
| Flatpak | The new bundle opens in your software app (Discover, GNOME Software): install it there, then start Clamor again. |
| Android (`.apk`) | Android asks "Update this app?": tap **Update**. The first time, Android may ask you to allow Clamor to install apps. |

The swarm keeps running until the update installs. A member device leaves the swarm properly
before restarting and rejoins on its own afterwards.

**Safety.** Every release has a manifest listing each file with its size and SHA-256 checksum,
signed with the Clamor release key. The app installs a file only if the manifest's signature is
valid for the key built into Clamor *and* the file matches the manifest exactly. A changed or
corrupted download is refused.

The `clamor-node` inside the app is updated together with the app, never on its own: running
`clamor-node update` on an app installation says to update the app instead.

## The command-line program

A standalone `clamor-node` (from a `clamor-<version>-*.tar.gz` or `.zip`) updates itself:

- `clamor-node update --check`: is there a newer release?
- `clamor-node update`: download, verify (same signed manifest) and install it; the previous
  version is kept next to it.
- `clamor-node update --rollback`: put the previous version back.
- `--auto-update` on `coordinator` or `join`: install new releases by itself while no model is
  running. `--no-update-check` turns the daily check off.

## Whole swarm

When a release changes how devices talk to each other (the release notes say so), update every
device: devices on different versions may not be able to work together.
