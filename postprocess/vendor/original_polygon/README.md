# Original polygon runtime

`original_run_standalone.py` is an unmodified copy of the production
standalone implementation from:

`inference_backend/Dinov3_postprocess/external/atosyori-pipeline-dev/src/atosyori_postprocess/legacy/run_standalone.py`

Source SHA-256 at import: `e3f33f635e7b56b6190f88b681d76e8b519ca5c5ecb849617ed343558980ac9c`.

The modular pipeline invokes only its hidden
`__onefile_polygon_optimize` entry point.  The adapter in `production.py`
converts the legacy artifacts to current internal contracts; this vendor file
must not define or mutate the public result SQLite schema.
