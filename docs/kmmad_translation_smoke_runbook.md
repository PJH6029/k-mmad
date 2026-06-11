# K-MMAD translation smoke runbook

This runbook follows the approved deep-interview and ralplan artifacts for the first K-MMAD infra smoke.

## Binding gates

- Full MMAD download is required before translation-smoke completion.
- Do not reserve `1x H200` until full-data manifest and sanity-check report pass.
- Use the cheapest feasible MLXP/cluster path for data acquisition and sanity checks.
- Do not delete or overwrite existing dataset-path data.
- First smoke translates only QA text plus captions and reports skipped text fields.

## Local dry checks

```bash
uv run python scripts/kmmad_download.py --method archive_urls
uv run python scripts/kmmad_sanity_check.py --dataset tests/fixtures/mmad_sample --expected-rows 2 --allow-partial --report-out /tmp/kmmad_sanity_fixture.json
uv run python scripts/kmmad_translate_smoke.py --dataset tests/fixtures/mmad_sample --output-dir /tmp/kmmad_translate_fixture --sample-size 2 --mock
uv run python scripts/kmmad_run_record.py init --out /tmp/kmmad_run_record.json
uv run python scripts/kmmad_run_record.py validate /tmp/kmmad_run_record.json
```

## Cluster sequence

1. Commit and push local scaffolding.
2. In the cluster clone at `/mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad`, pull the committed SHA.
3. Acquire full MMAD into `/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/original/mmad` using the selected non-destructive method.
4. Write a download manifest under `original/manifests`.
5. Run sanity check and require a passing report before H200 inference.
6. Only after manifest and sanity pass, reserve `1x H200` for LLM inference/translation smoke.
7. Start the OpenAI-compatible LLM endpoint or batch inference command.
8. Run `kmmad_translate_smoke.py` without `--mock` against the selected sample.
9. Validate output and preserve untranslated-field report plus inspection examples.
10. Cancel GPU reservation when idle and record evidence.

## Notes on upstream MMAD download

The MMAD GitHub README documents Hugging Face download options and the Hugging Face dataset currently exposes separate archives plus `metadata.csv`, `mmad.json`, and `domain_knowledge.json`. Prefer the current Hugging Face file layout over relying on a stale `ALL_DATA.zip` path.

## Source-file contract

- `metadata.csv` is the operational row source for translation smoke because it is the flattened QA table and matches the configured `expected_rows = 39672`.
- `mmad.json` is still sanity-checked as the image-keyed source structure; the first smoke recorded `8366` image keys and `39670` nested conversation turns. Treat this JSON-vs-CSV delta as provenance evidence, not as a translation-row source for the current smoke.
- The sanity report must record both the selected `record_file` and the `mmad_json` image/turn counts so future full-translation work can revisit the source contract deliberately.

## Minimal server contract

`scripts/kmmad_minimal_openai_server.py` is a smoke-only OpenAI-compatible shim for `/v1/chat/completions`; it is not production inference infrastructure. Keep `--trust-remote-code` disabled unless a pinned, reviewed model revision requires it, and record any such exception in the run record.

## Proprietary endpoint / openai-oauth smoke

This section covers the second K-MMAD smoke path: endpoint-backed proprietary LLM translation instead of local GPU LLM serving.

### Binding scope

- Live endpoint smoke uses `openai-oauth` first.
- Real OpenAI API-key mode must share the same translation client path, but this pass only contract-tests it with a local/fake endpoint. Do not make a paid/live OpenAI API call unless a later task explicitly authorizes it.
- Do not expose `openai-oauth` or any OpenAI-compatible endpoint publicly. Bind to `127.0.0.1` inside the trusted pod.
- Do not persist Codex auth cache contents, API keys, bearer tokens, refresh tokens, or proxy tokens in git, run records, logs, translated artifacts, or local copied outputs.
- Do not reserve H200 for endpoint-only smoke. Prefer `production-storage-shell-1` or another non-GPU production path with the same mounted repo/dataset paths.

### Local contract checks

```bash
uv run python -m unittest tests/test_endpoint_auth_contract.py
uv run python scripts/kmmad_translate_smoke.py --dataset tests/fixtures/mmad_sample --output-dir /tmp/kmmad_endpoint_translate_fixture --sample-size 2 --mock
uv run python scripts/kmmad_translate_smoke.py --validate-only /tmp/kmmad_endpoint_translate_fixture/translation_smoke.jsonl --untranslated-report /tmp/kmmad_endpoint_translate_fixture/untranslated_fields.json
uv run python scripts/kmmad_run_record.py init --out /tmp/kmmad_endpoint_run_record.json
uv run python scripts/kmmad_run_record.py validate /tmp/kmmad_endpoint_run_record.json
```

The contract test starts a fake OpenAI-compatible endpoint and proves bearer-env API-key behavior without calling the real OpenAI API.

### Endpoint config modes

Openai-oauth proxy mode, no API key:

```toml
[inference]
endpoint_provider = "openai_oauth"
auth_mode = "none"
openai_compatible_base_url = "http://127.0.0.1:10531/v1"
model = "<model id from /v1/models>"
```

Real OpenAI API-key mode, contract-tested only in this pass:

```toml
[inference]
endpoint_provider = "openai"
auth_mode = "bearer_env"
api_key_env = "OPENAI_API_KEY"
openai_compatible_base_url = "https://api.openai.com/v1"
model = "<OpenAI chat-completions-capable model>"
```

The run record may store `auth_mode` and `api_key_env`, but never the value of the environment variable.

### MLXP openai-oauth smoke sequence

Use the existing production storage pod when it exposes the same paths and Codex auth surface. If a new pod is needed, use image `ghcr.io/pjh6029/snupi-personal-codex:20260414`; the cluster Codex home is `/root/work/.codex`.

Example cluster-side sequence:

```bash
cd /mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad
git fetch origin && git checkout <recorded-sha>
export CODEX_HOME=/root/work/.codex
# Install only ephemerally inside the pod when needed.
npx openai-oauth --host 127.0.0.1 --port 10531 > /tmp/kmmad-openai-oauth.log 2>&1 &
# Health/model discovery: record model ids/status only, not auth cache contents.
curl -sS http://127.0.0.1:10531/v1/models | tee /tmp/kmmad-openai-oauth-models.json
# Set config/base URL to http://127.0.0.1:10531/v1 and selected model id, then run smoke.
uv run python scripts/kmmad_translate_smoke.py --output-dir /mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/translated/smoke/<run-id> --sample-size 3
uv run python scripts/kmmad_translate_smoke.py --validate-only /mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/translated/smoke/<run-id>/translation_smoke.jsonl --untranslated-report /mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/translated/smoke/<run-id>/untranslated_fields.json
```

After the run, copy actual translated sample artifacts locally under ignored `outputs/`, verify SHA256 checksums against the remote files, and run secret-grep over all copied artifacts/logs/run records.
