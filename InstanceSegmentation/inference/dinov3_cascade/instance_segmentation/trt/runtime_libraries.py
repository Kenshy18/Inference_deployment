"""TensorRT vendor-library discovery for the family-owned backbone."""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys


def _collect_site_paths() -> list[str]:
    paths: list[str] = []
    candidates = [path for path in sys.path if path and "site-packages" in path]
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass
    for candidate in candidates:
        resolved = os.path.realpath(candidate)
        if resolved not in paths and os.path.isdir(resolved):
            paths.append(resolved)
    return paths


def _append_runtime_libs_to_env() -> None:
    """Expose venv TensorRT and CUDA vendor libraries to the process."""

    candidates: list[str] = []
    for site_path in _collect_site_paths():
        candidates.extend(
            [
                f"{site_path}/tensorrt_libs",
                f"{site_path}/nvidia/cudnn/lib",
                f"{site_path}/nvidia/cublas/lib",
                f"{site_path}/nvidia/cuda_runtime/lib",
            ]
        )
    configured = os.environ.get("TENSORRT_LIB_DIR")
    if configured:
        candidates.append(configured)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [path for path in candidates if os.path.isdir(path)]
    if current:
        parts.append(current)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _preload_vendor_libs() -> None:
    """Preload TensorRT/CUDA libraries when loader search paths are unreliable."""

    for site_path in _collect_site_paths():
        exact_paths = (
            os.path.join(site_path, "tensorrt_libs", "libnvinfer.so.10"),
            os.path.join(site_path, "tensorrt_libs", "libnvinfer_plugin.so.10"),
            os.path.join(site_path, "tensorrt_libs", "libnvonnxparser.so.10"),
        )
        patterns = (
            os.path.join(site_path, "nvidia", "cudnn", "lib", "libcudnn*.so.9"),
            os.path.join(site_path, "nvidia", "cublas", "lib", "libcublas*.so.12"),
            os.path.join(site_path, "nvidia", "cuda_runtime", "lib", "libcudart*.so*"),
        )
        for path in (
            *exact_paths,
            *(item for pattern in patterns for item in glob.glob(pattern)),
        ):
            if not os.path.isfile(path):
                continue
            try:
                ctypes.CDLL(path)
            except Exception:
                pass


__all__ = ["_append_runtime_libs_to_env", "_preload_vendor_libs"]
