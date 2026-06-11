set -euo pipefail
cd /mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad
export DATASET_ROOT=/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data
export CLUSTER_RUN_ID=general-vqa-gpu-smoke-20260611T153105Z-361a4eb
export CLUSTER_RUN_DIR="$DATASET_ROOT/runs/$CLUSTER_RUN_ID"
export CODEX_HOME=/root/work/.codex
# Stop local GPU model server; endpoint smoke should use proprietary Codex OAuth endpoint, not local GPU inference.
if [ -f "$CLUSTER_RUN_DIR/bllossom-local-llm.pid" ]; then
  oldpid=$(cat "$CLUSTER_RUN_DIR/bllossom-local-llm.pid" || true)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    kill "$oldpid" || true
    sleep 3
  fi
fi
mkdir -p "$CLUSTER_RUN_DIR"
{
  echo "cluster_run_id=$CLUSTER_RUN_ID"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "CODEX_HOME=$CODEX_HOME"
  echo "auth_file_present=$(test -f /root/work/.codex/auth.json && echo yes || echo no)"
  echo "node=$(node --version 2>/dev/null || true)"
  echo "npm=$(npm --version 2>/dev/null || true)"
  echo "codex=$(codex --version 2>/dev/null || true)"
  echo "openai_oauth_command=npx -y openai-oauth --host 127.0.0.1 --port 10531"
} > "$CLUSTER_RUN_DIR/openai-oauth-environment.txt"
if [ -f "$CLUSTER_RUN_DIR/openai-oauth.pid" ] && kill -0 "$(cat "$CLUSTER_RUN_DIR/openai-oauth.pid")" 2>/dev/null; then
  echo "openai-oauth already running pid=$(cat "$CLUSTER_RUN_DIR/openai-oauth.pid")"
else
  nohup env CODEX_HOME="$CODEX_HOME" npx -y openai-oauth --host 127.0.0.1 --port 10531 \
    > "$CLUSTER_RUN_DIR/openai-oauth.log" 2>&1 &
  echo $! > "$CLUSTER_RUN_DIR/openai-oauth.pid"
  echo "started_openai_oauth_pid=$(cat "$CLUSTER_RUN_DIR/openai-oauth.pid")"
fi
