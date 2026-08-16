# Parity-frozen polygon compatibility engine

`original_run_standalone.py` is the immutable numerical kernel retained from
the former standalone implementation:

`inference_backend/Dinov3_postprocess/external/atosyori-pipeline-dev/src/atosyori_postprocess/legacy/run_standalone.py`

Source SHA-256 at import: `e3f33f635e7b56b6190f88b681d76e8b519ca5c5ecb849617ed343558980ac9c`.

It is not a public pipeline, stage, CLI, fallback, or SQLite writer.  The
deployed `production.polygon` package loads its numerical functions only
through `production/polygon/runtime/phase1_runtime.py`, patches the frozen
Recall/DP contract, and validates all public artifacts afterward.  Keeping the
source immutable preserves accepted mask parity while the surrounding
orchestration, preparation, materialization, schema, and manifests are owned
by responsibility-specific Production modules.
