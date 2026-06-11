# K-MMAD Repo Instructions

This repository is for translating MMAD(https://github.com/jam-cc/MMAD) into Korean.

## Operational rules for this repo
See `OPERATIONAL_RULES.md`.

- Edit code/docs locally in this repository.
- GPU workloads like LLM inference, or evaluation must execute on the MLXP cluster.
- Use `uv` as the default Python environment, package, and command runner for local checks and cluster jobs when Python tooling is involved; record exceptions and resolved environments in run records.
- Default workflow:
  1. local edit and small local checks;
  2. commit and push;
  3. reserve an MLXP pod when cluster/GPU/data access is needed;
  4. in the cluster clone, pull the pushed commit;
  5. run the job from the recorded git SHA/config/image;
  6. preserve metrics/checkpoints/run records;
  7. cancel the reservation when GPU work will be idle.
- Cluster clone/PVC repo path:
  `/mnt/ddn/prod-runs/jeonghunpark/code/dokpamo/k-mmad`.
- K-MMAD dataset path in MLXP pods:
  `/mnt/ddn/prod-runs/jeonghunpark/data/dokpamo/k-mmad-data/`.
- You should manage all the data in the dataset path, including original MMAD or translated K-MMAD.
- Use `MLXP.md` and `docs/reproduction_spec/OPERATIONAL_RULES.md` for reservation payload details. `MLXP_ACCESS_TOKEN` is in `.env`; never commit secrets.
- Routine MLXP reservations/cancellations within this project are pre-authorized. Use the `mlxp-reservation-api` workflow when reserving, inspecting, or cancelling pods.
- Use custom public Docker Hub images for cluster jobs, default repo `docker.io/pjh6029/k-mmad`. Prefer immutable tags like `<role>-<yyyymmdd>-<gitsha>` and record image digest; do not use `latest` for evaluation jobs. You can use already-built images like `vllm` for LLM inference for translation.
- Do not use the MLXP `debug` namespace/project for this repo; production reservations must stay within one node.
- Reserve the smallest GPU count that keeps the job efficient. Use 1 GPU for smoke/data/Tiny checks and 2–4 H200 GPUs only when the path is multi-GPU ready. Cancel pods when the next expected work is local coding, docs, or long non-GPU inspection.
- Use W&B for non-trivial evaluation logs with entity `pjh6029-seoul-national-university` and project `k-mmad`; keep `WANDB_API_KEY` in `.env` only.