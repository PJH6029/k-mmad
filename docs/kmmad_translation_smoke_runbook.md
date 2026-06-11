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
