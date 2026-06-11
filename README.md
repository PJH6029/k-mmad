# K-MMAD

K-MMAD translates the MMAD industrial anomaly-detection benchmark into Korean and now also scaffolds Korean translation smokes for selected difficult general VQA benchmarks.

This repository currently contains first-pass infra-smoke tooling for:

1. planning/downloading the original MMAD dataset into the MLXP dataset path,
2. sanity-checking the downloaded data,
3. running a tiny QA+caption Korean translation smoke against an OpenAI-compatible LLM endpoint,
4. validating and recording smoke artifacts.
5. testing proprietary OpenAI-compatible endpoint translation via `openai-oauth` plus API-key contract checks;
6. general VQA translation smokes for MME-RealWorld, BLINK, MMMU-Pro, and MEGA-Bench through benchmark adapters, local LLM, and endpoint paths.

See `docs/kmmad_translation_smoke_runbook.md` for the operational runbook.
