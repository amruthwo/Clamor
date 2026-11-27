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

The coordinator also announces the swarm on your local network (mDNS, the way printers
announce themselves), so devices can find it without its address. `--no-mdns` turns that off.

## Adding a device

On the new device, run the printed command once:

```bash
clamor-node join 192.168.1.20 --passphrase harbor-velvet-crab-lantern-ninety
```

Or leave out the address and let the device find the swarm:

```bash
clamor-node join --passphrase harbor-velvet-crab-lantern-ninety
```

If there is more than one swarm on the network, name the one you want with `--swarm home`.

Add `--name "Your name"` on a first join: it is how the others see you in the swarm (see
[people and signing in](people.md)).

### With a one-time code instead

Rather than handing out the passphrase, you can give a new device a code that works once:
press **Add device** in the dashboard, or run `clamor-node invite` on the coordinator. You get a
six-digit code and the command to run:

```bash
clamor-node join 192.168.1.20 --code 067-944
```

The code works for one device, for ten minutes; three wrong tries cancel it. Like the
passphrase, it is never sent over the network. After that first join the device is remembered
exactly as if it had used the passphrase.

Anyone in the swarm can make a code for **one of their own devices** (it joins as them); only
admins can make one for **someone new**.

Capitals, spaces and dashes in the passphrase do not matter: `Harbor Velvet Crab Lantern Ninety`
works too. If you would rather not type it on the command line (where other users of the same
computer could see it), put it in the `CLAMOR_PASSPHRASE` environment variable instead.

### By asking, with no code at all

If someone who can add people (an admin, or the owner) is at hand, the new device can simply
ask:

```bash
clamor-node join --ask --name "Cal"
```

(or, in the app, press **Join** with the code left empty). The device shows a **check code**,
and the dashboard shows "*cal-thinkpad* wants to join as Cal" with a check code too. If the two
codes match, press **Approve**; otherwise **Decline**. Matching codes prove the dashboard is
approving *that* device, not something else on the network asking at the same moment. The
request waits up to ten minutes.

## Coming back

Once a device has joined, it remembers the swarm. From then on:

```bash
clamor-node join
```

looks on the network for a swarm it has joined before and rejoins it, without the passphrase.
Because it recognises the swarm itself, not its address, it does not matter if the coordinator
got a new address from your router. If the swarm cannot be found on the network, the device tries
the address it last reached it at. A device that has joined swarms before never joins an unknown
one by itself: for a new swarm, name it with `--swarm` or give its address.

If the connection drops, the device reconnects by itself, and looks for the swarm on the network
again if the coordinator has moved.

To see what a device remembers:

```bash
clamor-node swarms
```

On the coordinator this also lists the passphrase and the devices it has approved. Add `--scan`
to also list the swarms announced on the network right now.

## Leaving

- **For now:** stop `clamor-node` (Ctrl-C). The device finishes what it is doing and leaves; it
  can come back any time with `clamor-node join`.
- **For good:** in the app, **Settings → Change or leave swarm → Leave**: the swarm removes the
  device (its key is no longer approved), it finishes its work and leaves, and it forgets the
  swarm. On the command line, `clamor-node leave home` makes the device forget the swarm (an
  admin removes it from the swarm's side). Joining it again needs a new invitation or the
  passphrase. Add `--purge` to also delete the model pieces it downloaded for that swarm
  (pieces another swarm also uses are kept).
- **Removing someone else's device:** on **Swarm** or **Coord.**, tap the device and press
  **Remove from swarm** (admins). An offline device shows **Forget this device** instead, which
  takes it off the list. It finishes its work and leaves, and its key is no longer approved: it
  needs the passphrase to join again. To keep it out for good, also change the passphrase
  (restart the coordinator with `--passphrase "new words"`).
- **Deleting a swarm you coordinate:** `clamor-node leave home --delete-swarm` deletes its
  passphrase and approvals. Starting the coordinator again makes a new swarm that every device
  has to join afresh.

## Disk space

Devices that do not have a model download only the pieces (layers) they run, and keep them for
next time in Clamor's cache folder (`~/.cache/clamor/slices` on Linux). The cache stays under
32 GB by default: beyond that, the pieces used least recently are deleted. Change the limit with
`--cache-max-gb 10`, or move the folder with `--cache-dir`. `clamor-node swarms` shows how much
it holds. Deleting the folder is always safe.

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

- **Connections are encrypted.** Each device's connection to the coordinator is TLS, and the
  join handshake signs that very TLS session, so nothing can sit in the middle and relay it.
  The links that carry a model's work between devices are TLS too, and each end accepts only
  the certificate the coordinator told it to expect.

### Not yet

- The plain dashboard (port 7421) is not encrypted: use `https://<coordinator>:7424/` in a
  browser, or the app (see [people and signing in](people.md)). Model downloads between devices
  are not encrypted either, but only approved devices get them (signed requests).
- A QR code for adding phones comes with the phone app.
- Finding the swarm on the network is tested on Linux. Some networks (guest Wi-Fi, some
  corporate networks) block it; give the address instead.


## If something goes wrong

| Message | What it means |
|---|---|
| `first join: this needs the swarm's passphrase` | This device has not joined this swarm before. Add `--passphrase`. |
| `wrong passphrase` | The words do not match the swarm's. `clamor-node swarms` on the coordinator shows them. |
| `your request to join was declined` / `nobody approved the request to join in time` | Ask again while someone is at the dashboard, or use a code or the passphrase. |
| `wrong code` / `that code has expired or was already used` | Make a new one: **Add device** in the dashboard. |
| `this swarm's coordinator key changed since you joined` | The coordinator was reinstalled, its config folder was deleted, or something else is answering at its address. If you expected this, join again with `--passphrase`. |
| `several swarms on the network` | Name the one you want: `--swarm home`. |
| `no swarm found on the network` | The coordinator is not running, is started with `--no-mdns` or listens on one address only (`--bind`), or your network blocks discovery. Give its address. |
| `this device is not approved in this swarm` | It was removed from the swarm (or the swarm was deleted and started again). Join with the passphrase. |
| `none of the swarms this device joined is on the network` | To join a new swarm, name it with `--swarm` or give its address. |
