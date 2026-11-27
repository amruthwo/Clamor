# The desktop app

The Clamor app is a window and a tray icon around `clamor-node`: no terminal needed. It starts
or joins a swarm, shows the swarm's dashboard, and keeps running in the tray when you close the
window.

## Installing

Download the installer for your system from the
[latest release](https://github.com/amruthwo/Clamor/releases/latest):

| System | File |
|---|---|
| Linux, any distribution (including immutable ones like Bazzite and Silverblue) | `Clamor-<version>-<arch>.flatpak`: `flatpak install --user Clamor-*.flatpak` |
| Debian, Ubuntu | `Clamor_<version>_<arch>.deb` |
| Fedora, openSUSE | `Clamor-<version>-1.<arch>.rpm` |
| Linux, no install | `Clamor_<version>_amd64.AppImage` (x86_64 only) |
| macOS (Apple silicon) | `Clamor_<version>_aarch64.dmg` |
| Windows | `Clamor_<version>_x64-setup.exe` |

The macOS and Windows installers are not signed yet: macOS says the app "cannot be checked for
malicious software" (right-click the app, **Open**, then **Open** again), and Windows shows
"Windows protected your PC" (**More info**, **Run anyway**).

## First start

The app asks what this computer should do:

- **Start a swarm.** This computer becomes the coordinator. Give the swarm a name; Clamor makes up
  a passphrase (**Settings → Swarm passphrase**). Add other devices with **Add device** on the
  **Swarm** tab (a one-time code), or give them the passphrase.
- **Join a swarm.** The app lists the swarms it finds on your network. Pick one, give your name,
  type the code from **Add device** (or the passphrase), and press **Join**. Or leave the code
  empty to **ask**: the app shows a check code, and someone who runs the swarm approves your
  device under **Coord. → Needs you**. If nothing is found, type the coordinator's address.

Turn on **Start Clamor when I log in** to have it start with your computer, in the tray (also
in **Settings → This app**).

After that the app remembers what it does: next time it starts straight into the dashboard.
**Settings → Change or leave swarm** offers:

- **Join or start another swarm:** Clamor stops here and goes back to the first screen; the swarm
  is remembered, so rejoining needs no passphrase.
- **Leave** (a device that joined): it tells the swarm, finishes its work, leaves and forgets the
  swarm; coming back needs a new invitation.
- **Delete** (the computer that runs the swarm): the swarm ends for everyone; its passphrase and
  approvals are deleted.

Either can also delete the model files this device downloaded for that swarm.

## Where things are

- **Model files:** the app keeps them in its own folder (tray menu: **Open models folder**):
  `~/.local/share/clamor/models` on Linux, `~/Library/Application Support/clamor/models` on
  macOS, `%APPDATA%\clamor\models` on Windows. A model needs to be on only one device: the others
  fetch the parts they run.
- **Settings and keys:** the same config folder as the command-line `clamor-node` (see
  [joining](joining.md)), so the app and the command line share swarms.
- **The log:** **Settings → This app → Log**.
- **The layout:** Chat, Swarm, Models and Settings (plus Coord. for the owner and admins): a bar
  at the bottom on phones, a rail at the side on wider windows; on a wide screen the chat list
  stays beside the conversation, with the swarm's health and speed along the top.
- **Updates:** the app offers new versions itself and installs them when you agree; see
  [updating](updating.md).

## Known limits

- Tested by hand on Linux (Fedora Asahi, Apple M1: the .deb/.rpm build and the Flatpak, run as
  a coordinator). The macOS and Windows installers are built by CI but not yet tried on real
  machines.
- The Flatpak cannot start Clamor at login yet.
- On a desktop without a tray (or without the tray library), the app runs without a tray icon,
  and closing the window quits Clamor.
- On a member device the app is signed in as you automatically (see [people and signing in](people.md)).
