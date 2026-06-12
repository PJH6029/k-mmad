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

## General VQA multi-benchmark smoke

This section covers the first-pass general VQA extension for exactly these benchmarks:

- MME-RealWorld
- BLINK
- MMMU-Pro
- MEGA-Bench

### Scope boundary

The MMAD full-download and H200 gate above applies to the original MMAD smoke only. It is **not** the completion gate for this General VQA first pass.

General VQA first-pass success is sample/smoke based:

- no full benchmark translation;
- no benchmark evaluation, leaderboard submission, or model comparison;
- no adapter implementation beyond the four selected benchmarks;
- no destructive dataset operations;
- no paid/live real OpenAI API-key call;
- no requirement to fully download all four benchmarks before declaring the adapter/sample smoke complete.

Mock output is sanity-only. Completion evidence must include both actual local GPU LLM translation samples and actual `openai-oauth` endpoint translation samples.

### Local checks

```bash
uv run python scripts/kmmad_translate_smoke.py --list-benchmarks
uv run python scripts/kmmad_translate_smoke.py \
  --dataset tests/fixtures/general_vqa \
  --benchmarks mme_realworld,blink,mmmu_pro,mega_bench \
  --output-dir /tmp/kmmad_general_vqa_mock \
  --sample-size 1 \
  --mock \
  --concurrency 2
uv run python scripts/kmmad_translate_smoke.py --validate-only /tmp/kmmad_general_vqa_mock
uv run python -m unittest tests/test_general_vqa_adapters.py
```

### Artifact layout

Preferred durable cluster layout:

```text
/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/
  original/general-vqa/<benchmark>/...
  translated/smoke/general-vqa/<run-id>/<benchmark>/
    translation_smoke.jsonl
    untranslated_fields.json
    translation_validation.json
    inspection_examples.json
    translation_smoke_summary.json
  runs/general-vqa/<run-id>/run_record.json
```

Local copies of actual translated sample artifacts must be downloaded under ignored `outputs/`, with sanitized run evidence under `run_records/` when appropriate. Verify remote/local SHA256 checksums and run secret scans before reporting completion.

General VQA benchmark loading is fail-closed: each selected benchmark must expose one of its registered record filenames under its benchmark directory (for example `blink/blink.json`). The runner must not satisfy `blink` from an unrelated parseable file in the dataset root.

Directory-level `--validate-only` checks exact benchmark coverage from `general_vqa_translation_summary.json` and the full evidence bundle, not just discovered JSONL rows: each summary-listed benchmark must have `translation_smoke.jsonl`, `untranslated_fields.json`, `translation_validation.json`, `inspection_examples.json`, and `translation_smoke_summary.json`, and no unexpected benchmark output directory should be present.

### Local GPU LLM evidence path

1. Run local adapter/mock/API-contract checks first.
2. Commit and push the verified code.
3. Reserve the smallest appropriate production MLXP GPU pod, usually 1x H200 for this smoke.
4. In the cluster clone, pull the recorded git SHA.
5. Start the selected local OpenAI-compatible LLM endpoint or approved local inference shim.
6. Translate 1-3 sample records per selected benchmark.
7. Validate outputs, preserve run evidence, copy artifacts locally, and cancel the reservation promptly.
8. Record the resolved image digest or an archived image manifest summary in the final run record.

### openai-oauth endpoint evidence path

Use image `ghcr.io/pjh6029/snupi-personal-codex:20260414` and `CODEX_HOME=/root/work/.codex` when Codex auth is required.

```bash
export CODEX_HOME=/root/work/.codex
npx -y openai-oauth --host 127.0.0.1 --port 10531 > /tmp/kmmad-general-vqa-openai-oauth.log 2>&1 &
curl -sS http://127.0.0.1:10531/v1/models > /tmp/kmmad-general-vqa-models.json
uv run python scripts/kmmad_translate_smoke.py \
  --dataset /mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/original/general-vqa \
  --benchmarks mme_realworld,blink,mmmu_pro,mega_bench \
  --output-dir /mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/translated/smoke/general-vqa/<run-id> \
  --sample-size 1 \
  --concurrency 2
```

Bind `openai-oauth` only to `127.0.0.1`; do not expose a public proxy. Real OpenAI API-key mode remains contract-tested only in this pass: no paid/live OpenAI API call is allowed without a later explicit task.

### Provider/auth regression expectations

- `endpoint_provider = "openai_oauth"` must use no Authorization header by default.
- `auth_mode = "openai_oauth"` legacy alias, if accepted, must normalize to no-auth behavior.
- `auth_mode = "bearer_env"` must send `Authorization: Bearer $OPENAI_API_KEY` only to a fake/local contract endpoint in this pass.
- Persist env var names, not secret values.

## HF dataset export / repackaging pipeline

This section covers the post-translation path for later model evaluation. It assumes that full translation and translation quality control have already been completed upstream; the exporter records the declared QC status but does not score translation quality.

### Contract

Input:

- one or more existing `translation_smoke.jsonl` bundles, or a multi-benchmark directory with `general_vqa_translation_summary.json`;
- source/provenance metadata such as the original HF dataset ID, config, and revision;
- a declared translation-QC status (`unchecked`, `passed`, `failed`, or `needs_review`).

Output:

```text
<export-dir>/
  dataset/                    # Hugging Face DatasetDict.save_to_disk artifact
  data/<split>.jsonl           # normalized review/debug shards
  README.md                    # dataset card for future Hub/manual upload
  hf_export_manifest.json      # provenance, split counts, no-upload proof
  hf_export_validation.json    # load_from_disk/count/record-id/secret-scan validation
```

Rows keep source IDs, benchmark IDs, split/task metadata, media references, answers/preserve fields, skipped fields, original source JSON, and translated Korean text fields. Heterogeneous nested structures are stored as JSON strings to keep Arrow columns stable across MMAD and general VQA benchmarks. MMAD multi-turn rows include `conversation_index` in the derived source ID so `record_id` remains unique across QA turns from the same image.

The implementation follows current Hugging Face `datasets` API guidance: create in-memory datasets from normalized dictionaries, then persist/reload with `DatasetDict.save_to_disk()` and `datasets.load_from_disk()` for local evaluation reuse. `--hub-repo-id` records an intended future target and dry-run `push_to_hub` command in the manifest only; it never uploads. The exporter redacts secret-like values before writing normalized rows, stores row-level translation artifact provenance as relative labels such as `blink/translation_smoke.jsonl`, and validates the manifest, dataset card, every `data/*.jsonl` shard, and loaded dataset rows for secret-like values and duplicate/missing `record_id`s.

### Local export smoke

```bash
# Produce a small translated fixture bundle first.
uv run python scripts/kmmad_translate_smoke.py \
  --dataset tests/fixtures/general_vqa \
  --benchmarks blink,mmmu_pro \
  --output-dir /tmp/kmmad-general-vqa-translated \
  --sample-size 1 \
  --mock \
  --concurrency 2

# Repackage into HF datasets format.
uv run python scripts/kmmad_hf_export.py \
  --translation-output /tmp/kmmad-general-vqa-translated \
  --output-dir /tmp/kmmad-general-vqa-hf-export \
  --dataset-name k-general-vqa-ko \
  --original-hf-dataset fixture/general-vqa \
  --original-hf-config adapter-fixture \
  --translation-qc-status passed \
  --hub-repo-id pjh6029/k-general-vqa-ko

# Validate the saved artifact can be loaded and matches the manifest.
uv run python scripts/kmmad_hf_export.py --validate-only /tmp/kmmad-general-vqa-hf-export
```

### Safety gates

- The exporter refuses to write into a non-empty output directory; use a fresh export path for every run.
- It validates upstream translation bundles by default before packaging.
- It redacts secret-like values before row persistence and scans the manifest, dataset card, every split JSONL shard, and loaded HF dataset rows during validation.
- It records QC as a declared upstream status. Production/release-ready exports require `translation_qc_status=passed` plus a machine-readable `--translation-qc-report`; otherwise the manifest/card should be treated as a non-release packaging artifact.
- Directory inputs are fail-closed: an export directory must either be a direct translation bundle containing `translation_smoke.jsonl` or a multi-benchmark bundle with `general_vqa_translation_summary.json`; nested stale folders are not auto-discovered.
- It does not call `push_to_hub`, make paid API calls, reserve GPUs, mutate source datasets, or delete existing dataset-path data.
