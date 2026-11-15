# Joining a swarm

A swarm has one **coordinator** (the computer you started it on) and any number of devices that
join it. Joining takes one command the first time, and none of your attention after that.

## Starting a swarm

```bash
clamor-node coordinator --swarm home
```

The first time, Clamor makes up a **passphrase** of five random words and prints it, with the
command other devices need:

```
swarm 'home' passphrase: harbor-velvet-crab-lantern-ninety
add a device: clamor-node join 192.168.1.20 --passphrase harbor-velvet-crab-lantern-ninety
```

The swarm keeps that passphrase from then on: restarting the coordinator does not change it.
To choose your own, start with `--passphrase "your own words"` (this also changes the passphrase
of an existing swarm; devices that already joined are not affected).

## Adding a device

On the new device, run the printed command once:

```bash
clamor-node join 192.168.1.20 --passphrase harbor-velvet-crab-lantern-ninety
```

Capitals, spaces and dashes in the passphrase do not matter: `Harbor Velvet Crab Lantern Ninety`
works too. If you would rather not type it on the command line (where other users of the same
computer could see it), put it in the `CLAMOR_PASSPHRASE` environment variable instead.

## Coming back

Once a device has joined, it remembers the swarm. From then on:

```bash
clamor-node join
```

rejoins the swarm it joined last, without the passphrase. If the connection drops, the device
reconnects by itself.

To see what a device remembers:

```bash
clamor-node swarms
```

On the coordinator this also lists the passphrase and the devices it has approved.

## How it stays safe

- **The passphrase never crosses the network.** The device and the coordinator use it to agree on a
  secret (a "password-authenticated key exchange", SPAKE2) and prove to each other that they both
  know it. Someone listening on the network learns nothing they could use to guess it.
- **Every device has its own key.** On the first join the coordinator approves the device's key,
  and the device remembers the coordinator's key. After that they recognise each other by these
  keys, so the passphrase is only needed once.
- **A fake coordinator is noticed.** If something else answers at the coordinator's address, a
  device that joined before refuses it unless you give the passphrase again.
- **Models are shared only within the swarm.** A device serves its model files only to requests
  signed by an approved device.
- **A device has a different identity in every swarm.** Two swarms cannot tell that they have the
  same device in common.

Keys are kept in Clamor's config folder (`~/.config/clamor` on Linux,
`~/Library/Application Support/clamor` on macOS, `%APPDATA%\clamor` on Windows), readable only by
you: `device.key` is the device's secret, and `swarms/` holds one file per swarm.

### Not yet

- The connections are not encrypted yet: someone on your network could read what devices send each
  other (prompts and replies included), although they cannot join or pretend to be a device.
- The dashboard has no login: anyone on your network can open it.
- Removing a single device's approval, finding the swarm without typing an address, and adding a
  device with a one-time code or QR code from the dashboard are coming next.

Only run Clamor on a network you trust until these are done.

## If something goes wrong

| Message | What it means |
|---|---|
| `first join: this needs the swarm's passphrase` | This device has not joined this swarm before. Add `--passphrase`. |
| `wrong passphrase` | The words do not match the swarm's. `clamor-node swarms` on the coordinator shows them. |
| `this swarm's coordinator key changed since you joined` | The coordinator was reinstalled, its config folder was deleted, or something else is answering at its address. If you expected this, join again with `--passphrase`. |
| `this device has not joined a swarm yet` | `clamor-node join` without an address needs a swarm joined before. |
