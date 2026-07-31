"""Translate a registered model into its isolated standalone CLI process."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from contracts import TaskType
from registry import ModelRegistration
from progress_protocol import (
    INTERVAL_ENVIRONMENT,
    PHASE_ENVIRONMENT,
    emit_phase_progress,
)

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
        backend = resolve_backend(registration, request.face_backend)
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
        if registration.backend_cli_argument is not None:
            command.extend([registration.backend_cli_argument, backend])
        if request.face_trt_bundle is not None:
            command.extend(
                [
                    "--trt-bundle",
                    str(request.face_trt_bundle.expanduser().resolve()),
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
    phase = (
        "segmentation_inference"
        if invocation.role == "instance_segmentation"
        else "face_inference"
    )
    environment[PHASE_ENVIRONMENT] = phase
    environment.setdefault(INTERVAL_ENVIRONMENT, "0.3")
    emit_phase_progress(
        phase,
        state="running",
        completed=0,
        total=None,
        detail="model-loading",
    )
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


def execute_invocations_parallel(
    invocations: tuple[ModelInvocation, ...] | list[ModelInvocation],
    *,
    stagger_seconds: float = 0.0,
) -> None:
    """Execute isolated model CLIs concurrently and fail as one atomic group."""

    if not invocations:
        return
    if stagger_seconds < 0:
        raise ValueError("stagger_seconds must be non-negative")
    ordered = list(invocations)
    if stagger_seconds > 0:
        ordered.sort(key=lambda item: item.role != "face_detection")
    processes: dict[subprocess.Popen[bytes], ModelInvocation] = {}
    try:
        for index, invocation in enumerate(ordered):
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            phase = (
                "segmentation_inference"
                if invocation.role == "instance_segmentation"
                else "face_inference"
            )
            environment[PHASE_ENVIRONMENT] = phase
            environment.setdefault(INTERVAL_ENVIRONMENT, "0.3")
            emit_phase_progress(
                phase,
                state="running",
                completed=0,
                total=None,
                detail="model-loading",
            )
            print(
                f"[orchestrator] parallel role={invocation.role} "
                f"model={invocation.registration.model_id} "
                f"backend={invocation.backend}",
                flush=True,
            )
            process = subprocess.Popen(
                invocation.command,
                cwd=invocation.working_directory,
                env=environment,
            )
            processes[process] = invocation
            if stagger_seconds > 0 and index + 1 < len(ordered):
                deadline = time.monotonic() + stagger_seconds
                while time.monotonic() < deadline:
                    return_code = process.poll()
                    if return_code is not None:
                        if return_code != 0:
                            raise RuntimeError(
                                f"{invocation.registration.model_id} inference "
                                f"failed with exit code {return_code} before "
                                "the sibling model was started"
                            )
                        break
                    time.sleep(min(0.05, deadline - time.monotonic()))

        failure: tuple[ModelInvocation, int] | None = None
        while processes:
            for process, invocation in tuple(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del processes[process]
                if return_code != 0:
                    failure = (invocation, return_code)
                    break
            if failure is not None:
                break
            if processes:
                time.sleep(0.05)

        if failure is not None:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            invocation, return_code = failure
            raise RuntimeError(
                f"{invocation.registration.model_id} inference failed "
                f"with exit code {return_code}; sibling models were stopped"
            )
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    for invocation in invocations:
        if not invocation.output_path.is_file():
            raise RuntimeError(
                f"model did not create SQLite output: {invocation.output_path}"
            )


__all__ = [
    "ModelInvocation",
    "build_invocation",
    "execute_invocation",
    "execute_invocations_parallel",
    "resolve_backend",
]
