# People, roles and signing in

A swarm is shared by **people**, each with their own devices. Everyone who joins becomes a
person in the swarm; a device you add with **Add device → One of my devices** joins as *you*.

## Roles

| Role | Who | Can |
|---|---|---|
| **Owner** | Whoever started the swarm (the coordinator's computer) | Everything, including making people admins |
| **Admin** | Appointed by the owner | Add people, remove people and devices, start and stop models, re-run the muster |
| **Member** | Everyone else | Use the swarm (chat), see the dashboard, add their own devices, change their own devices' settings |

The **People** section of the dashboard lists everyone, their role and their devices. The owner
can make someone an admin (or a member again); admins can remove people. Removing someone makes
all their devices leave, ends their sign-ins, and they need a new invitation to come back.

Every device in the device list shows whose it is. You can change the settings of your own
devices (how much memory and CPU Clamor may use, when it pauses); only admins can change other
people's.

Admins also see devices **asking to join** (see [joining](joining.md)): approve one only if it
shows the same check code as the dashboard.

## Chats

The **Home** page has a **Group** chat that everyone in the swarm shares (each message shows who
wrote it), and your own conversations (**New chat**), which only you can open, from any of your
devices. Conversations are kept on the coordinator, so whoever runs the coordinator's computer
could read the files; chats that even the owner cannot read are planned.

## Signing in

You rarely need to:

- **On the coordinator's computer** the dashboard is always the owner's.
- **In the Clamor app on any other device** you are already signed in as yourself: the app
  shows the swarm's dashboard through the device's own connection to the swarm, which is
  encrypted and signed with the device's key. (`clamor-node join` serves the same dashboard at
  `http://127.0.0.1:7421/` on that device.)

**In a browser** on another computer, phone or tablet, open the dashboard
(`http://<coordinator>:7421/`). It shows a six-digit code and waits. Then either:

- **approve it** from a device where you are signed in: the dashboard there shows "A browser
  wants to sign in", with the code. Check the code matches, and press **Approve as me**. The
  browser signs in as you. (Only approve codes you see on your own screen.)
- or open **Sign in with your PIN instead**, with your name and PIN. Set a PIN in the People
  section (**Set my PIN**). After 5 wrong PINs, PIN sign-in is locked for 15 minutes.

A browser stays signed in for 30 days, until you press **Sign out**, or until you are removed.

## Encryption

The connections between devices, and the app's dashboard on member devices, are encrypted. The
plain browser dashboard on port 7421 is not: someone on your network could see what it shows.
For an encrypted dashboard in a browser, open `https://<coordinator>:7424/` instead. Your browser
will warn that it does not know the certificate (Clamor makes its own; no certificate authority
is involved): accept it once for that address. The certificate stays the same while the swarm
exists.

## If you lose access

Whoever can run Clamor on the coordinator's computer is the owner: the dashboard there needs no
sign-in, and `clamor-node swarms` there shows the passphrase. There is no way to lock the owner
out of their own swarm.
