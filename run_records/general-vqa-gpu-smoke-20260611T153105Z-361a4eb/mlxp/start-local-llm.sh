set -euo pipefail
cd /mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad
export DATASET_ROOT=/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data
export CLUSTER_RUN_ID=general-vqa-gpu-smoke-20260611T153105Z-361a4eb
export CLUSTER_RUN_DIR="$DATASET_ROOT/runs/$CLUSTER_RUN_ID"
export SAMPLE_ROOT="$DATASET_ROOT/original/general-vqa"
export TRANSLATED_ROOT="$DATASET_ROOT/translated/smoke/$CLUSTER_RUN_ID/local-llm"
export HF_HOME="$DATASET_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$CLUSTER_RUN_DIR" "$SAMPLE_ROOT" "$TRANSLATED_ROOT" "$HF_HOME"
# Non-destructive sample staging: these are tiny committed fixtures for smoke evidence, not full benchmark downloads.
rsync -a tests/fixtures/general_vqa/ "$SAMPLE_ROOT/"
find "$SAMPLE_ROOT" -maxdepth 2 -type f | sort > "$CLUSTER_RUN_DIR/sample-files.txt"
cat > "$CLUSTER_RUN_DIR/local-gpu-config.toml" <<'EOF'
[paths]
dataset_root = "/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data"
original_subdir = "original/mmad"
general_vqa_original_subdir = "original/general-vqa"
manifest_subdir = "original/manifests"
translated_subdir = "translated/smoke"
run_subdir = "runs"

[source]
hf_dataset = "jiang-cc/MMAD"
hf_url = "https://huggingface.co/datasets/jiang-cc/MMAD"
github_url = "https://github.com/jam-cc/MMAD"
arxiv_url = "https://arxiv.org/abs/2410.09453"
expected_rows = 39672
expected_images = 8366
expected_total_size_gb = 28.4
preferred_methods = ["git_lfs_clone", "huggingface_cli_download", "archive_urls"]
archive_files = ["DS-MVTec.zip", "GoodsAD.zip", "MVTec-AD.zip", "MVTec-LOCO.zip", "VisA.zip", "domain_knowledge.json", "metadata.csv", "mmad.json"]

[smoke]
sample_size = 8
sample_seed = 6029
concurrency = 1
translate_fields = ["question", "options", "caption"]
preserve_fields = ["answer", "image", "image_path", "id", "metadata", "category", "dataset", "split"]
report_untranslated_text_fields = true

[inference]
endpoint_provider = "local_openai_compatible"
auth_mode = "none"
api_key_env = "OPENAI_API_KEY"
openai_compatible_base_url = "http://127.0.0.1:18000/v1"
model = "Qwen/Qwen2.5-1.5B-Instruct"
timeout_seconds = 180
max_tokens = 256
temperature = 0.0

[mlxp]
project = "production"
member = "jeonghunpark"
purpose_prefix = "SKT Dokpamo - MMAD Translation"
cluster_repo_path = "/mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad"
h200_gate = "General VQA sample smoke: full MMAD download gate is not applicable."

[inference.endpoint_auth]
redact_headers = ["Authorization"]

[general_vqa]
benchmarks = ["mme_realworld", "blink", "mmmu_pro", "mega_bench"]
fixture_root = "tests/fixtures/general_vqa"
first_pass_full_download_required = false
mock_is_completion_evidence = false
requires_local_gpu_llm_evidence = true
requires_openai_oauth_evidence = true

[general_vqa.parallelism]
default_concurrency = 2
deterministic_order = true
EOF
{
  echo "cluster_run_id=$CLUSTER_RUN_ID"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "model=Qwen/Qwen2.5-1.5B-Instruct"
  echo "server_url=http://127.0.0.1:18000/v1"
  echo "HF_HOME=$HF_HOME"
  echo "python=$(python -V 2>&1)"
  echo "torch=$(python -c 'import torch; print(torch.__version__)')"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$CLUSTER_RUN_DIR/local-llm-environment.txt"
# Start actual local model inference server. Use system Python to preserve CUDA torch; dependency install logged separately.
if [ -f "$CLUSTER_RUN_DIR/local-llm.pid" ] && kill -0 "$(cat "$CLUSTER_RUN_DIR/local-llm.pid")" 2>/dev/null; then
  echo "server already running pid=$(cat "$CLUSTER_RUN_DIR/local-llm.pid")"
else
  nohup env PYTHONUNBUFFERED=1 HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" CUDA_VISIBLE_DEVICES=0 \
    python scripts/kmmad_minimal_openai_server.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --host 127.0.0.1 \
      --port 18000 \
    > "$CLUSTER_RUN_DIR/local-llm-server.log" 2>&1 &
  echo $! > "$CLUSTER_RUN_DIR/local-llm.pid"
  echo "started pid=$(cat "$CLUSTER_RUN_DIR/local-llm.pid")"
fi
