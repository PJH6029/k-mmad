set -euo pipefail
cd /mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad
export DATASET_ROOT=/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data
export CLUSTER_RUN_ID=general-vqa-gpu-smoke-20260611T153105Z-361a4eb
export CLUSTER_RUN_DIR="$DATASET_ROOT/runs/$CLUSTER_RUN_ID"
export SAMPLE_ROOT="$DATASET_ROOT/original/general-vqa"
export TRANSLATED_ROOT="$DATASET_ROOT/translated/smoke/$CLUSTER_RUN_ID/openai-oauth"
cat > "$CLUSTER_RUN_DIR/openai-oauth-config.toml" <<'EOF'
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
endpoint_provider = "openai_oauth"
auth_mode = "none"
api_key_env = "OPENAI_API_KEY"
openai_compatible_base_url = "http://127.0.0.1:10531/v1"
model = "gpt-5.4"
timeout_seconds = 180
max_tokens = 256
temperature = 0.0

[mlxp]
project = "production"
member = "jeonghunpark"
purpose_prefix = "SKT Dokpamo - MMAD Translation"
cluster_repo_path = "/mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad"
h200_gate = "General VQA endpoint smoke: openai-oauth Codex image path, no API key."

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
  echo "## openai-oauth endpoint translation smoke"
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "git_sha=$(git rev-parse HEAD)"
  echo "endpoint_provider=openai_oauth"
  echo "base_url=http://127.0.0.1:10531/v1"
  echo "model=gpt-5.4"
  curl -fsS http://127.0.0.1:10531/v1/models | python -m json.tool
  PYTHONPATH=scripts python - <<'PY'
from kmmad_translate_smoke import build_request_headers
config={"inference":{"endpoint_provider":"openai_oauth","auth_mode":"none","openai_compatible_base_url":"http://127.0.0.1:10531/v1","model":"gpt-5.4"}}
print("request_headers="+str(build_request_headers(config)))
PY
  python scripts/kmmad_translate_smoke.py \
    --config "$CLUSTER_RUN_DIR/openai-oauth-config.toml" \
    --dataset "$SAMPLE_ROOT" \
    --output-dir "$TRANSLATED_ROOT" \
    --benchmarks mme_realworld,blink,mmmu_pro,mega_bench \
    --sample-size 1 \
    --concurrency 1
  echo "## validate directory"
  python scripts/kmmad_translate_smoke.py --validate-only "$TRANSLATED_ROOT"
} 2>&1 | tee "$CLUSTER_RUN_DIR/openai-oauth-translation.log"
