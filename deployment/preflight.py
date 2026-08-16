#!/usr/bin/env python3
"""Deployment gate for a prepared WSL clone."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--profile", choices=("core", "all"), default="core")
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(
            os.environ.get(
                "INFERENCE_RUNTIME_PYTHON",
                "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
            )
        ),
    )
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    runtime_python = args.runtime_python.expanduser().resolve()
    failures: list[str] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "root": str(root),
        "profile": args.profile,
        "runtime_python": str(runtime_python),
    }
    pruned_manifest = Path("/opt/mask-pipeline/release/pruned-development-paths.txt")
    if pruned_manifest.is_file():
        expected_absent = [
            root / line.strip()
            for line in pruned_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        remaining = [str(path) for path in expected_absent if path.exists()]
        report["runtime_only_tree"] = {
            "manifest": str(pruned_manifest),
            "pruned_paths": len(expected_absent),
            "remaining": remaining,
        }
        if remaining:
            failures.append(f"development-only paths remain: {remaining}")
    if not (root / ".git").exists():
        failures.append(f"not a Git worktree root: {root}")
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        failures.append(f"runtime Python is unavailable: {runtime_python}")
    try:
        report["commit"] = run(["git", "rev-parse", "HEAD"], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"Git commit check failed: {exc}")
    try:
        command = [
            str(runtime_python),
            str(root / "deployment" / "verify_assets.py"),
            "--root",
            str(root),
            "--profile",
            args.profile,
            "--stage",
            "runtime",
        ]
        if args.full_hash:
            command.append("--full-hash")
        report["assets"] = run(command, cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"asset validation failed: {exc}")
    try:
        probe = run(
            [
                str(runtime_python),
                "-c",
                (
                    "import json,orchestration,tensorrt,torch;"
                    "print(json.dumps({'torch':torch.__version__,"
                    "'tensorrt':tensorrt.__version__,"
                    "'cuda':torch.version.cuda,"
                    "'cuda_available':torch.cuda.is_available(),"
                    "'gpu':torch.cuda.get_device_name(0),"
                    "'capability':torch.cuda.get_device_capability(0)}))"
                ),
            ],
            cwd=root,
        )
        runtime = json.loads(probe)
        report["python_runtime"] = runtime
        if runtime.get("cuda_available") is not True:
            failures.append("PyTorch cannot access CUDA")
        if runtime.get("capability") != [12, 0]:
            failures.append(
                f"production engines require SM120; found {runtime.get('capability')}"
            )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        failures.append(f"Python/GPU probe failed: {exc}")
    try:
        postprocess_probe = run(
            [
                str(runtime_python),
                "-c",
                (
                    "import json,sys;"
                    f"sys.path.insert(0,{str(root / 'postprocess')!r});"
                    "import run_pipeline,nms.production,production.polygon;"
                    "from common.registry import stage_implementations;"
                    "forbidden_modules={'nms.adaptive','nms.component_aware',"
                    "'nms.stages'};"
                    "loaded=sorted(forbidden_modules.intersection(sys.modules));"
                    "assert not loaded,loaded;"
                    "p=run_pipeline.build_parser();"
                    "options=sorted(o for a in p._actions for o in a.option_strings);"
                    "retired_options=sorted(set(options).intersection({"
                    "'--shape-mode','--device','--max-gap','--model-root',"
                    "'--k2-run-dir','--k2-batch-size'}));"
                    "assert not retired_options,retired_options;"
                    "a=p.parse_args(['--input-jsonl','input.jsonl',"
                    "'--output-dir','output']);"
                    "stages=[s.implementation for s in "
                    "run_pipeline._configured_pipeline(a).stages];"
                    "assert 'nms.production_v3' in stages,stages;"
                    "assert 'production.polygon_v3_cpu' in stages,stages;"
                    "registered=stage_implementations();"
                    "retired_stages=sorted(set(registered).intersection({"
                    "'approximation.polygon.rdp','keyframes.polygon.interval',"
                    "'gap_fill.polygon.linear','approximation.ellipse.production',"
                    "'keyframes.ellipse.dense','gap_fill.ellipse.linear'}));"
                    "assert not retired_stages,retired_stages;"
                    "print(json.dumps({'stages':stages,'retired_options':"
                    "retired_options,'retired_stages':retired_stages,"
                    "'loaded_forbidden':loaded}))"
                ),
            ],
            cwd=root,
        )
        report["postprocess_runtime"] = json.loads(postprocess_probe)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        failures.append(f"Production postprocess import probe failed: {exc}")
    native_build = root / "postprocess/production/polygon/runtime/native_interval/build"
    try:
        native_probe = run(
            [
                str(runtime_python),
                "-c",
                (
                    "import json,sys;"
                    f"sys.path.insert(0,{str(native_build)!r});"
                    "import native_interval_metrics as n;"
                    "v=n.exact_metrics([[[0,0],[4,0],[4,4],[0,4]]],"
                    "[[[0,0],[4,0],[4,4],[0,4]]]);"
                    "print(json.dumps({'module':n.__file__,'iou':v['iou'],"
                    "'recall':v['recall']}))"
                ),
            ],
            cwd=root,
        )
        report["native_polygon_evaluator"] = json.loads(native_probe)
        if report["native_polygon_evaluator"].get("iou") != 1.0:
            failures.append("native polygon evaluator returned an invalid IoU")
        if report["native_polygon_evaluator"].get("recall") != 1.0:
            failures.append("native polygon evaluator returned an invalid Recall")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        failures.append(f"native polygon evaluator probe failed: {exc}")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        failures.append("nvidia-smi is unavailable inside WSL")
    else:
        try:
            report["nvidia_smi"] = run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
                cwd=root,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(f"nvidia-smi probe failed: {exc}")
    overlay_binary = root / "overlay" / "native" / "build" / "overlay_native"
    overlay_ffmpeg = (
        root / "overlay" / "native" / ".runtime" / "ffmpeg" / "bin" / "ffmpeg"
    )
    fast_overlay_ffmpeg = (
        root / "overlay" / ".runtime" / "ffmpeg-nvenc-btbn-8.1" / "bin" / "ffmpeg"
    )
    fast_overlay_ffprobe = fast_overlay_ffmpeg.with_name("ffprobe")
    report["overlay"] = {
        "binary": str(overlay_binary),
        "ffmpeg": str(overlay_ffmpeg),
        "fast_ffmpeg": str(fast_overlay_ffmpeg),
        "fast_ffprobe": str(fast_overlay_ffprobe),
    }
    for path in (
        overlay_binary,
        overlay_ffmpeg,
        fast_overlay_ffmpeg,
        fast_overlay_ffprobe,
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            failures.append(f"overlay runtime is unavailable: {path}")
    output = root / "output"
    output.mkdir(exist_ok=True)
    if not os.access(output, os.W_OK):
        failures.append(f"output directory is not writable: {output}")
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
