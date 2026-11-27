# Models

## Finding one

**Models → Find a model** lists the models Clamor knows about, and for each one how it would run
on *your* swarm right now: whether it fits, and roughly how fast (tokens per second), split over
how many devices. The predictions come from the swarm's measurements (the muster), so they
change as devices join, leave or pause.

- **Best quality / Smaller:** the Q8 file (best quality; what Clamor is tested with) or the Q4
  file (about half the size, so bigger models fit; should work, less tested).
- **Filters:** *Fits this swarm*, *Runs in Clamor*, and what a model can do: **Tools** (it can
  call functions), **Thinking** (it reasons step by step before answering), **Vision**
  (pictures), **Embedding** (turns text into vectors for search; not a chat model), **Code**.
  Several can be on at once.
- **Best fit** marks the biggest model that runs here at a usable speed.

Tap a model for its details: maker, licence, size, context length, the exact file and where it
comes from.

## Downloading

Admins see **Download** (or **Start**, once the swarm has the model). The file goes to the
coordinator's models folder; the other devices fetch only the layers they run, as for any model.
Downloads continue if you close the page, resume where they stopped if interrupted, and are
checked against the size and SHA-256 in Clamor's catalog (which ships inside Clamor's signed
releases): a changed or broken file is refused.

You can still add any `.gguf` file yourself: put it in the models folder of any device
(desktop app: tray menu, **Open models folder**).

## Managing models

**Models → On the swarm's devices** lists every model file in the swarm. Tap one to see which
devices hold it, start it, or delete it (admins):

- **Delete** next to a device removes the file from that device only (other copies stay).
- **Delete from all devices** removes the file everywhere, and also the parts of it that other
  devices cached while running it, so the disk space really comes back.

A model that is running cannot be deleted: stop it first. Devices older than Clamor 0.7.0 cannot
be asked to delete, and are listed when that happens.

## What Clamor runs today

Clamor's engine runs the **Llama** and **Qwen2** families (Llama 3.x, Qwen2.5, SmolLM2 and
similar). Models with other architectures or tokenizers (Qwen3, Gemma 3, Mistral's newer
models, the DeepSeek R1 Qwen distills, embedding models) are listed with the reason they do not
run yet, so you can see what is coming. Vision, embeddings and showing a model's "thinking"
nicely need more engine work.

## The catalog

The list is built by `tools/build-catalog.py` from the real files on Hugging Face: their sizes
and checksums, and the facts in each file's header (architecture, tokenizer, layers, context
length, whether its chat template supports tools or thinking). Only the name, maker, licence and
a line about each model are written by hand.
