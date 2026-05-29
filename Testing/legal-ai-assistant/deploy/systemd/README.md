# Auto-start Conan on boot (systemd user services)

Two **user** services run the backend and the ngrok tunnel and keep them up
(auto-restart on failure). No root is needed to install or enable them — the
only root step is enabling *linger* so they also start at boot **before** you
log in.

- `conan-backend.service` — `uvicorn app.main:app` on `:8000`
- `conan-ngrok.service` — `ngrok http --url=pushiness-jumble-policy.ngrok-free.dev 8000`,
  gated on the backend being healthy first.

> Paths in the unit files are absolute and specific to this machine/user
> (`/home/marno000onaaa/...`). Edit them if you relocate the project or run as
> another user.

## Install / enable

```bash
mkdir -p ~/.config/systemd/user
cp conan-backend.service conan-ngrok.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now conan-backend.service conan-ngrok.service
```

## Make it start at boot without logging in (one-time, needs sudo)

```bash
sudo loginctl enable-linger "$USER"
```

Without linger, the services start when you log in; with linger, they start at
boot. Either way the reserved ngrok URL comes back the same.

## Everyday commands

```bash
systemctl --user status  conan-backend.service conan-ngrok.service
systemctl --user restart conan-ngrok.service     # e.g. after a network blip
systemctl --user stop    conan-backend.service conan-ngrok.service
journalctl --user -u conan-backend.service -f    # live backend logs
journalctl --user -u conan-ngrok.service   -f    # live tunnel logs
```

The reserved public URL is **https://pushiness-jumble-policy.ngrok-free.dev**
(send header `ngrok-skip-browser-warning: true`). See `../../../backend_contract.md`.
