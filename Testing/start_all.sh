#!/usr/bin/env bash
# =============================================================================
# start_all.sh — bring up the Conan backend + the stable ngrok tunnel in one go.
#
#   ./start_all.sh
#
# Idempotent: it stops any previous backend/ngrok first, then starts fresh,
# waits until each is actually serving, and prints the public URL.
# Re-run it any time (after a reboot, laptop wake, or a crash) — the ngrok URL
# is a reserved domain, so it comes back the SAME every time.
# =============================================================================

set -u

# ── Config (override via env if needed) ──────────────────────────────────────
PORT="${CONAN_PORT:-8000}"
NGROK_DOMAIN="${NGROK_DOMAIN:-pushiness-jumble-policy.ngrok-free.dev}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-240}"   # seconds to wait for the backend to load models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/legal-ai-assistant"
PY="$APP_DIR/LaW/bin/python"
NGROK="$HOME/bin/ngrok"; command -v ngrok >/dev/null 2>&1 && NGROK="$(command -v ngrok)"
BACKEND_LOG="$APP_DIR/backend.log"
NGROK_LOG="$APP_DIR/ngrok.log"

say(){ printf '\n\033[1;36m%s\033[0m\n' "$*"; }
ok(){  printf '\033[1;32m✅ %s\033[0m\n' "$*"; }
err(){ printf '\033[1;31m❌ %s\033[0m\n' "$*"; }

# ── Sanity checks ────────────────────────────────────────────────────────────
[ -x "$PY" ]    || { err "Backend venv python not found at $PY"; exit 1; }
[ -x "$NGROK" ] || { err "ngrok not found at $NGROK (install it or set PATH)"; exit 1; }

# ── 1. Stop anything already running ─────────────────────────────────────────
say "🧹 Stopping any previous backend (port $PORT) and ngrok agent..."
fuser -k "${PORT}/tcp" 2>/dev/null
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f "ngrok http"          2>/dev/null
sleep 2

# ── 2. Start the backend ─────────────────────────────────────────────────────
say "🚀 Starting FastAPI backend on :$PORT ..."
( cd "$APP_DIR" && nohup "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" > "$BACKEND_LOG" 2>&1 & )

printf "   waiting for the index + models to load (up to %ss)" "$HEALTH_TIMEOUT"
deadline=$(( SECONDS + HEALTH_TIMEOUT ))
until curl -sf -m 5 "http://localhost:${PORT}/api/v1/health" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo; err "Backend did not become healthy in ${HEALTH_TIMEOUT}s. Check $BACKEND_LOG"; exit 1
  fi
  printf "."; sleep 3
done
echo; ok "Backend healthy at http://localhost:${PORT}"

# ── 3. Start the ngrok tunnel (reserved static domain) ───────────────────────
say "🌍 Starting ngrok tunnel → https://${NGROK_DOMAIN} ..."
nohup "$NGROK" http --url="$NGROK_DOMAIN" "$PORT" --log=stdout > "$NGROK_LOG" 2>&1 &

# wait until the tunnel is registered or an error appears
for _ in $(seq 1 20); do
  if grep -qiE 'ERR_NGROK|authentication failed' "$NGROK_LOG" 2>/dev/null; then
    echo; err "ngrok failed to authenticate/connect — check $NGROK_LOG"
    grep -iE 'ERR_NGROK|authentication failed' "$NGROK_LOG" | head -1; exit 1
  fi
  grep -q "started tunnel" "$NGROK_LOG" 2>/dev/null && break
  sleep 1
done

# verify it actually serves through the public URL
if curl -sf -m 25 -H 'ngrok-skip-browser-warning: true' \
        "https://${NGROK_DOMAIN}/api/v1/health" >/dev/null 2>&1; then
  ok "Tunnel live and serving the API"
else
  err "Tunnel started but the public URL didn't answer yet — give it a few seconds, then check: curl https://${NGROK_DOMAIN}/api/v1/health"
fi

# ── 4. Summary ───────────────────────────────────────────────────────────────
say "🏛️  Conan is up."
cat <<EOF
   • Public API (backend team): https://${NGROK_DOMAIN}/api/v1
   • Swagger UI:                 https://${NGROK_DOMAIN}/docs
   • Local API:                  http://localhost:${PORT}/api/v1
   • Logs:                       $BACKEND_LOG   |   $NGROK_LOG

   Leave this machine awake to keep the URL reachable.
   To stop everything:  pkill -f 'uvicorn app.main:app'; pkill -f 'ngrok http'
EOF
