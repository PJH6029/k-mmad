# K-MMAD

K-MMAD translates the MMAD industrial anomaly-detection benchmark into Korean and now also scaffolds Korean translation smokes for selected difficult general VQA benchmarks.

This repository currently contains first-pass infra-smoke tooling for:

1. planning/downloading the original MMAD dataset into the MLXP dataset path,
2. sanity-checking the downloaded data,
3. running a tiny QA+caption Korean translation smoke against an OpenAI-compatible LLM endpoint,
4. validating and recording smoke artifacts.
5. testing proprietary OpenAI-compatible endpoint translation via `openai-oauth` plus API-key contract checks;
6. general VQA translation smokes for MME-RealWorld, BLINK, MMMU-Pro, and MEGA-Bench through benchmark adapters, local LLM, and endpoint paths;
7. exporting translated/QC-passed artifacts into Hugging Face `datasets` format for later model evaluation (`DatasetDict.save_to_disk`, JSONL shards, manifest, and dataset card).

See `docs/kmmad_translation_smoke_runbook.md` for the operational runbook.


## HF dataset export

After full translation and translation QC, package translation artifacts into a Hugging Face-compatible dataset directory:

```bash
uv run python scripts/kmmad_hf_export.py \
  --translation-output /path/to/translated/run \
  --output-dir /path/to/hf-export/k-mmad-ko \
  --dataset-name k-mmad-ko \
  --original-hf-dataset jiang-cc/MMAD \
  --translation-qc-status passed
uv run python scripts/kmmad_hf_export.py --validate-only /path/to/hf-export/k-mmad-ko
```

The exporter does not upload to the Hub or delete existing data. It writes `dataset/` (loadable with `datasets.load_from_disk`), `data/*.jsonl`, `hf_export_manifest.json`, `hf_export_validation.json`, and a dataset-card `README.md`; validation scans manifest/card/every JSONL shard/loaded dataset rows for secret-like values and duplicate record IDs, while row-level source artifact provenance is stored as portable relative labels.
