# Connecting apps (OpenAI-compatible API)

Clamor speaks the same API as OpenAI's chat models, so apps made for it can use your swarm's
model instead: coding assistants, chat apps, scripts and agent frameworks.

## What you need

1. **A model running** in the swarm (**Models**).
2. **The address:** in Clamor, **Settings → Connect apps → API keys** shows it, for example
   `http://192.168.1.40:7421/v1`. Apps on the coordinator computer, or on a device running the
   Clamor app, can also use `http://127.0.0.1:7421/v1`.
3. **An API key:** on the same screen, give the key a name and press **Create**. It is shown once:
   copy it into the app. A key acts as you, only for this API, and you can delete it any time.
   (Apps on your own device through `127.0.0.1` need no key; if an app insists on one, any text
   works there.)

The model name does not matter: the swarm runs one model, and every request goes to it.
`GET /v1/models` lists it.

## Examples

**curl**

```bash
curl http://192.168.1.40:7421/v1/chat/completions \
  -H "Authorization: Bearer clm_..." -H "Content-Type: application/json" \
  -d '{"model": "clamor", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Python** (OpenAI's own library)

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.40:7421/v1", api_key="clm_...")
reply = client.chat.completions.create(model="clamor", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

**Apps:** look for an "OpenAI-compatible" or "custom OpenAI" provider and set its base URL (also
called API base or endpoint) to the address above and the key to your Clamor key. For example:
Continue (provider `openai` with `apiBase`), Aider (`--openai-api-base` and `--model
openai/<model>`), Open WebUI (**Settings → Connections → OpenAI API**). The exact settings differ
between apps and versions; their documentation names them.

## What works

- `GET /v1/models` and `POST /v1/chat/completions`, with or without streaming;
  `max_tokens` / `max_completion_tokens`, `stop`, and usage (`stream_options.include_usage`).
- Text messages from the `system`, `developer`, `user` and `assistant` roles.

Not yet:

- **Tool calls** (functions): `tools` is ignored and the model answers in text. Coding agents
  that rely on tools work only in their plain chat or edit modes.
- **Sampling per request** (`temperature`, `top_p`, ...): the running model's settings apply.
- Images, embeddings, and `n` greater than 1.

## Good to know

- On the plain address (port 7421) your key crosses your network unencrypted. Apps that accept
  self-signed certificates can use `https://<coordinator>:7424/v1` instead.
- Requests from apps share the swarm with people chatting: long replies keep it busy.
