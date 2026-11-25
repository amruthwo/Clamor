# Private chats

A private chat keeps your words on your own device, even though the swarm does most of the
work.

## How it works

A language model is a long stack of layers. Your words only exist as words at the two ends of
that stack: on the way in (the first layers turn them into numbers) and on the way out (the last
layers turn numbers back into words). Everything in between is long lists of numbers.

In a private chat, **your device runs both ends**: the first and the last eighth of the model.
The swarm's other devices, the coordinator included, run the layers in between, with the parts
of the model they already have loaded. They only ever see numbers. The conversation itself is
kept on your device, not on the coordinator.

## Using it

Private chats are available in the **Clamor app on a member device** (or in a browser on that
device at `http://127.0.0.1:7421/`), where your own Clamor runs. In the chat list, under
**Private · this device**, press **+ New private chat**. The first time, press **Turn on private
chats**: your device loads its part of the model (a few hundred megabytes for a small model,
more for a large one), which takes a moment. After that, private chats work like the others:
replies appear as they are written, and **Stop** ends one early.

A model has to be running in the swarm first. If the model changes, or a device running part of
your private chat leaves, the chat says so; turn private chats on again.

On the coordinator's own computer there is no private option: the coordinator *is* that
computer, so your personal chats already stay on it.

## What it protects, and what it does not

**Protected:** what you write and what the model answers. It is never sent anywhere as text,
and never stored anywhere but on your device. The swarm's owner cannot read it in the
coordinator's files, because it is not there.

**Not hidden:**

- *That* you chatted, when, and roughly how long the messages were: the other devices see the
  numbers go by.
- The numbers in the middle are not encryption. Researchers have shown that, with the same
  model and deliberate effort, numbers from a model's layers can sometimes be turned back into
  approximate text. Running more layers on your device makes this harder; Clamor runs an eighth
  of the model at each end.
- Your device's own files: anyone with access to your device can read its private chats.

So: private from the other people in the swarm, including whoever runs it, unless someone makes
a deliberate technical effort to reconstruct it.

## Good to know

- Your device needs room for its two ends of the model, and does the heaviest per-word work at
  the very end of the model, so a private chat can be a little slower than a normal one.
- Private chats are kept per swarm on each device. They do not (yet) follow you to your other
  devices.
