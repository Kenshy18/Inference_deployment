"""Translate a registered model into its isolated standalone CLI process."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from contracts import TaskType
from registry import ModelRegistration

from .config import OrchestrationRequest


INFERENCE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    registration: ModelRegistration
    role: str
    backend: str
    output_path: Path
    command: tuple[str, ...]
    working_directory: Path


def resolve_backend(
    registration: ModelRegistration,
    requested: str,
) -> str:
    backend = (
        registration.default_backend if requested == "auto" else requested
    )
    if backend not in registration.backends:
        raise ValueError(
            f"{registration.model_id} does not support backend {backend!r}; "
            f"available={list(registration.backends)}"
        )
    return backend


def build_invocation(
    registration: ModelRegistration,
    *,
    role: str,
    output_path: Path,
    request: OrchestrationRequest,
) -> ModelInvocation:
    family = registration.package.split(".", 1)[0]
    working_directory = INFERENCE_ROOT / family
    script = working_directory / "infer.py"
    if not script.is_file():
        raise FileNotFoundError(f"model inference CLI not found: {script}")
    runtime_python = request.runtime_python.expanduser().resolve()
    if not runtime_python.is_file():
        raise FileNotFoundError(f"runtime Python not found: {runtime_python}")
    if role == "instance_segmentation":
        if registration.task is not TaskType.INSTANCE_SEGMENTATION:
            raise ValueError(
                f"{registration.model_id} is not a segmentation model"
            )
        backend = resolve_backend(
            registration, request.segmentation_backend
        )
    elif role == "face_detection":
        if registration.task is not TaskType.OBJECT_DETECTION:
            raise ValueError(f"{registration.model_id} is not a detector")
        backend = registration.default_backend
    else:
        raise ValueError(f"unsupported model role: {role}")

    command = [
        str(runtime_python),
        str(script),
        "--input",
        str(request.input_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
        "--device",
        request.device,
        "--overwrite",
        "--fast-sqlite",
    ]
    if request.max_frames is not None:
        command.extend(["--max-frames", str(request.max_frames)])
    if role == "instance_segmentation":
        command.extend(["--warmup-frames", str(request.warmup_frames)])
        if registration.backend_cli_argument is not None:
            command.extend([registration.backend_cli_argument, backend])
    else:
        command.extend(
            [
                "--warmup-iterations",
                str(request.face_warmup_iterations),
                "--progress-interval",
                "0",
            ]
        )
        if request.face_classes:
            command.append("--classes")
            command.extend(request.face_classes)
    return ModelInvocation(
        registration=registration,
        role=role,
        backend=backend,
        output_path=output_path,
        command=tuple(command),
        working_directory=working_directory,
    )


def execute_invocation(invocation: ModelInvocation) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(
        f"[orchestrator] role={invocation.role} "
        f"model={invocation.registration.model_id} "
        f"backend={invocation.backend}",
        flush=True,
    )
    try:
        subprocess.run(
            invocation.command,
            cwd=invocation.working_directory,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{invocation.registration.model_id} inference failed "
            f"with exit code {exc.returncode}"
        ) from exc
    if not invocation.output_path.is_file():
        raise RuntimeError(
            f"model did not create SQLite output: {invocation.output_path}"
        )


__all__ = [
    "ModelInvocation",
    "build_invocation",
    "execute_invocation",
    "resolve_backend",
]
