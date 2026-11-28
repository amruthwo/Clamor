# Clamor on Android

The Android app is the same app as on desktops: join a swarm (or start one), see the swarm's
dashboard, chat, and lend the phone's memory and computing power to the swarm.

> **Early.** The Android app is new and has not yet been tried on many phones. It needs Android 11
> or newer and a 64-bit ARM phone (practically every phone from the last several years).

## Installing

Download `Clamor-<version>-android-arm64.apk` from the
[latest release](https://github.com/amruthwo/Clamor/releases/latest) on the phone, open it, and
allow your browser or file manager to install apps when Android asks ("Install unknown apps").
Clamor is not in the Play Store yet. Later versions are offered by the app itself (see
[updating](updating.md)).

## First start

The same first screen as on desktops: **Join a swarm** (the app lists the swarms it finds on your
Wi-Fi; type the code from **Add device**, the passphrase, or nothing to ask an admin) or **Start a
swarm**. Allow notifications when asked: the notification is what lets Clamor stay in your swarm
while the app is in the background.

## Joining by QR code

On the first screen, **Scan a QR code** and point the camera at the code from **Add device** on a
device that is already in the swarm. Clamor asks for the camera only then. The scanner is
Clamor's own, so it also works on phones without Google Play services (GrapheneOS, LineageOS,
/e/OS). You can always type the six-digit code instead.

## When the phone helps

A phone is a **phone** to Clamor: by default it takes work only while it is **charging** and you
are not using it, and it uses at most half its memory and CPU. Change this in **Devices →
Settings** in the dashboard. While charging, the app keeps the phone awake enough to keep
working (for example overnight); on battery it lets the phone sleep.

Models it runs live in the app's own storage. A phone usually fetches just its share of a model
from another device in the swarm rather than holding whole models.

## Good to know

- The notification ("Part of your swarm…") stays while Clamor runs. Closing the app from the
  recent-apps list stops it; open it again to rejoin.
- Private chats work on the phone like on a computer: the phone runs both ends of the model.
- Finding the swarm on Wi-Fi needs a network that allows it (most home networks do; some guest
  networks do not). Otherwise type the coordinator's address.
- The app cannot start by itself when the phone starts, yet.
