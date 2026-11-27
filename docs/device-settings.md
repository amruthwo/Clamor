# Device settings

Every device in a swarm decides how much of itself Clamor may use. The defaults are chosen so
that lending your computer to the swarm never gets in your way: Clamor uses a share of the
machine, runs at low priority, and steps aside when you need the device.

## The settings

| Setting | What it does |
|---|---|
| **Take part** | **Full**: always takes work. **Idle only**: steps aside while you are using the device and comes back when it is quiet. **Paused**: takes no new work. |
| **Work on battery** | Off: a laptop or phone pauses whenever it is unplugged, and resumes when you plug it back in. |
| **Memory Clamor may use** | A share of the device's RAM. Clamor also always leaves 512 MB free, whatever the share. |
| **CPU Clamor may use** | A share of the device's fast cores. One fast core always stays free for you. |
| **Priority** | **Low**: your own apps always get the CPU first, so the device never feels slow. **Normal**: Clamor competes with them equally. Either way, only Clamor's model computation is affected: its connection to the swarm always runs at normal priority, so a busy device never looks disconnected. |

"Pausing" never interrupts a reply that is already being written: it finishes, and then the
device takes no new requests until it is available again.

## Renaming a device

Tap a device (on **Swarm**) and the pencil next to its name. The new name is what everyone in the
swarm sees; it is kept by the swarm, so it stays when the device restarts. You can rename your
own devices; admins can rename any. Underneath, the device keeps its id (its host name, or the
name of a phone), which the command line and the logs use. Leave the name empty to show the id
again.

## Defaults

| | Desktops | Laptops, tablets and phones |
|---|---|---|
| Take part | Full | Idle only |
| Work on battery | – | Off |
| Memory | 70% | 50% |
| CPU | 80% | 50% |
| Priority | Low | Low |

Clamor tells laptops from desktops by whether the device has a battery.

## Changing them

**In the dashboard** (any browser on your network, `http://<coordinator>:7421/`): press
**Settings** next to a device. The panel shows what each setting means for that device, for
example "60% · 9.4 of 15.6 GB" or "2 of 4 fast cores", along with its battery and free memory.

**On the device itself**, when starting Clamor:

```bash
clamor-node join <coordinator> --participation full --allow-battery --ram-share 0.6 --cpu-share 0.8
clamor-node join <coordinator> --priority normal
```

The same options work for `clamor-node coordinator`. Options given on the command line win;
otherwise a device uses the settings it was last given from the dashboard *in that swarm*;
otherwise the defaults.

## When changes take effect

- **Take part, work on battery:** within a few seconds.
- **Memory, CPU and priority:** from the next pipeline (a model that is already loaded keeps
  running as it was set up). No restart is needed on any system.

## Messages you may see

- **"stage 0 on laptop needs 27.4 GB … but may use 8.2 GB"** when starting a model: that part of
  the model does not fit in the memory this device may use. Give the device fewer layers, use a
  smaller context (`ctx`), or raise its memory share. Clamor checks this *before* loading
  anything, so a device is never pushed into swapping.
- **"paused: laptop is not taking work right now (on battery …)"** when chatting: one of the
  devices running the model is paused. Plug it in, change its settings, or wait until it is free.

## Where settings are kept

Each device remembers its settings **per swarm**, in its own config folder, so they survive
restarts, and a device can be generous in your home swarm and frugal in a friend's. When a
device joins, it tells the coordinator its settings for that swarm, so they also survive the
coordinator being restarted or replaced.

- Linux: `~/.config/clamor/swarms/settings-<swarm>.json`
- macOS: `~/Library/Application Support/clamor/swarms/settings-<swarm>.json`
- Windows: `%APPDATA%\clamor\swarms\settings-<swarm>.json`

Settings saved by Clamor 0.4 (`device-settings.json`) are used for any swarm that has none of
its own yet. Command-line options apply in every swarm. Use `--config-dir` to keep all of this
somewhere else.

## Who can change them

A device's settings can be changed by the person it belongs to, and by the swarm's admins and
owner (see [people and signing in](people.md)).
