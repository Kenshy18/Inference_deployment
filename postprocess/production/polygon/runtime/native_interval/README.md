# Production native interval evaluator

This pybind11 module implements the exact raster metrics, interval batches,
native DP helpers, and exact pair-vote batches used by the promoted polygon
post-processor. It is an optimization of the validated CPU arithmetic, not an
approximate CUDA path.

The deployment setup builds and validates it automatically:

```bash
INFERENCE_RUNTIME_PYTHON=/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  postprocess/production/polygon/runtime/native_interval/bootstrap_and_build.sh
```

The build environment is stored below
`/home/kenshin/.local/share/video-mask-runtime/native-interval`; the extension
itself is written to this directory's ignored `build/` tree. The WSL
distribution therefore contains both the extension and its private OpenCV
runtime, while the Git repository contains the reproducible source and build
contract.

`bootstrap_and_build.sh` runs the full generated-coordinate parity suite and
then imports the resulting module with the actual Production Python. The
deployment preflight repeats a minimal exact-metric probe and refuses to ship
an image if the module cannot be loaded.
