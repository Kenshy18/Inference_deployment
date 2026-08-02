import { execFile, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import type { AppSettings } from "../shared/types";

const execFileAsync = promisify(execFile);
const WSL_TIMEOUT_MS = 20_000;

export const WSL_RUNNER_SOURCE = `#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


STAGING_ROOT = Path("/tmp/mask-pipeline-studio")
RUNNER_MARKER = ".wsl-runner.json"
LEGACY_STALE_SECONDS = 24 * 60 * 60


def _process_start_ticks(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        suffix = text.rsplit(")", 1)[1].split()
        return suffix[19]
    except (IndexError, OSError, ValueError):
        return None


def _process_matches(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        pid = int(value["pid"])
        expected = str(value["start_ticks"])
    except (KeyError, TypeError, ValueError):
        return False
    return pid > 1 and _process_start_ticks(pid) == expected


def _write_marker(path: Path, child_pid: int | None = None) -> None:
    runner_pid = os.getpid()
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": time.time(),
        "runner": {
            "pid": runner_pid,
            "start_ticks": _process_start_ticks(runner_pid),
        },
    }
    if child_pid is not None:
        payload["child"] = {
            "pid": child_pid,
            "start_ticks": _process_start_ticks(child_pid),
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _legacy_directory_in_use(path: Path) -> bool:
    token = os.fsencode(str(path))
    try:
        process_roots = Path("/proc").iterdir()
    except OSError:
        return True
    for process_root in process_roots:
        if not process_root.name.isdigit():
            continue
        try:
            command = (process_root / "cmdline").read_bytes()
        except OSError:
            continue
        if token in command:
            return True
    return False


def _cleanup_stale_staging(current: Path | None) -> None:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for candidate in STAGING_ROOT.iterdir():
        if not candidate.is_dir() or candidate == current:
            continue
        marker = candidate / RUNNER_MARKER
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            if _process_matches(payload.get("runner")) or _process_matches(
                payload.get("child")
            ):
                continue
            shutil.rmtree(candidate, ignore_errors=True)
            continue
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age >= LEGACY_STALE_SECONDS and not _legacy_directory_in_use(candidate):
            shutil.rmtree(candidate, ignore_errors=True)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: wsl-runner.py PID_FILE COMMAND [ARG ...]", file=sys.stderr)
        return 2
    pid_file = Path(sys.argv[1])
    arguments = sys.argv[2:]
    sync_from = None
    sync_to = None
    sync_pending = None
    staging_job = None
    marker = None
    if arguments and arguments[0] == "--sync-output":
        if len(arguments) < 4:
            print("--sync-output requires SOURCE TARGET and COMMAND", file=sys.stderr)
            return 2
        sync_from = Path(arguments[1])
        sync_to = Path(arguments[2])
        arguments = arguments[3:]
    command = arguments
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    staging_root = STAGING_ROOT.resolve()
    if sync_from is not None:
        resolved_sync_from = sync_from.resolve()
        if staging_root in resolved_sync_from.parents:
            staging_job = resolved_sync_from.parent
    _cleanup_stale_staging(staging_job)
    if staging_job is not None:
        staging_job.mkdir(parents=True, exist_ok=True)
        marker = staging_job / RUNNER_MARKER
        _write_marker(marker)

    process = None

    def forward(signum: int, _frame: object) -> None:
        if process is None:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        if marker is not None:
            _write_marker(marker, process.pid)
        temporary = pid_file.with_suffix(pid_file.suffix + ".tmp")
        temporary.write_text(f"{process.pid}\\n", encoding="ascii")
        os.replace(temporary, pid_file)
        code = process.wait()
        if code == 0 and sync_from is not None and sync_to is not None:
            print(f"[gui-sync] copying {sync_from} -> {sync_to}", flush=True)
            sync_to.parent.mkdir(parents=True, exist_ok=True)
            copy_target = sync_to
            if not sync_to.exists():
                sync_pending = sync_to.with_name(sync_to.name + ".partial")
                shutil.rmtree(sync_pending, ignore_errors=True)
                copy_target = sync_pending
            shutil.copytree(sync_from, copy_target, dirs_exist_ok=True)
            source_text = str(sync_from)
            target_text = str(sync_to)
            for json_path in copy_target.rglob("*.json"):
                try:
                    text = json_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                replaced = text.replace(source_text, target_text)
                if replaced == text:
                    continue
                temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
                temporary_json.write_text(replaced, encoding="utf-8")
                os.replace(temporary_json, json_path)
            if sync_pending is not None:
                os.replace(sync_pending, sync_to)
            print("[gui-sync] Windows output copy complete", flush=True)
        return code
    finally:
        if sync_from is not None:
            try:
                resolved_sync_from = sync_from.resolve()
                if staging_root in resolved_sync_from.parents:
                    shutil.rmtree(resolved_sync_from, ignore_errors=True)
                    if marker is not None:
                        marker.unlink(missing_ok=True)
                    try:
                        resolved_sync_from.parent.rmdir()
                    except OSError:
                        # Keep a shared/non-empty parent; only the per-job
                        # empty staging directory is eligible for removal.
                        pass
            except OSError:
                pass
        if sync_pending is not None:
            shutil.rmtree(sync_pending, ignore_errors=True)
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
`;

export function windowsToWslPath(value: string): string {
  const trimmed = value.trim();
  const uncMatch =
    /^\\\\wsl(?:\.localhost)?\\([^\\]+)\\(.*)$/i.exec(trimmed);
  if (uncMatch) {
    return `/${uncMatch[2].replaceAll("\\", "/")}`;
  }
  const driveMatch = /^([a-zA-Z]):[\\/](.*)$/.exec(trimmed);
  if (driveMatch) {
    return `/mnt/${driveMatch[1].toLowerCase()}/${driveMatch[2].replaceAll("\\", "/")}`;
  }
  return trimmed.replaceAll("\\", "/");
}

export function wslToWindowsPath(value: string, distro: string): string {
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  const driveMatch = /^\/mnt\/([a-zA-Z])(?:\/(.*))?$/.exec(normalized);
  if (driveMatch) {
    const suffix = driveMatch[2]?.replaceAll("/", "\\") ?? "";
    return `${driveMatch[1].toUpperCase()}:\\${suffix}`;
  }
  if (normalized.startsWith("/")) {
    return `\\\\wsl.localhost\\${distro}${normalized.replaceAll("/", "\\")}`;
  }
  return value;
}

export function isWslPathInside(value: string, root: string): boolean {
  const candidate = path.posix.normalize(value.replaceAll("\\", "/"));
  const base = path.posix.normalize(root.replaceAll("\\", "/"));
  return candidate === base || candidate.startsWith(`${base}/`);
}

function distroArgs(settings: AppSettings): string[] {
  return settings.wslDistro.trim()
    ? ["-d", settings.wslDistro.trim()]
    : [];
}

async function runWsl(
  settings: AppSettings,
  args: string[],
  cwd?: string,
): Promise<{ stdout: string; stderr: string }> {
  const result = await execFileAsync(
    "wsl.exe",
    [
      ...distroArgs(settings),
      ...(cwd ? ["--cd", cwd] : []),
      "--",
      ...args,
    ],
    {
      timeout: WSL_TIMEOUT_MS,
      windowsHide: true,
      maxBuffer: 8 * 1024 * 1024,
      encoding: "utf8",
    },
  );
  return { stdout: result.stdout, stderr: result.stderr };
}

export async function validateWslBackend(settings: AppSettings): Promise<void> {
  if (settings.backendMode !== "wsl" || process.platform !== "win32") {
    return;
  }
  if (!settings.wslDistro.trim()) {
    throw new Error("WSL distributionを設定してください。");
  }
  const backendRoot = windowsToWslPath(settings.backendRoot);
  const runtimePython = windowsToWslPath(settings.runtimePython);
  try {
    await runWsl(
      settings,
      [
        "/usr/bin/test",
        "-d",
        backendRoot,
        "-a",
        "-x",
        runtimePython,
      ],
      backendRoot,
    );
    await runWsl(
      settings,
      [runtimePython, "-c", "import orchestration; print('ok')"],
      backendRoot,
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `WSLバックエンドを確認できません。distribution、リポジトリroot、実行Pythonを確認してください。\n${detail}`,
    );
  }
}

export function launchPidPath(windowsJobDir: string): string {
  return path.join(windowsJobDir, "wsl-process.pid");
}

export function wslLaunchWrapper(
  settings: AppSettings,
  pidPath: string,
  runnerPath: string,
  command: string[],
  syncOutput?: { source: string; target: string },
): string[] {
  return [
    ...distroArgs(settings),
    "--cd",
    windowsToWslPath(settings.backendRoot),
    "--",
    windowsToWslPath(settings.runtimePython),
    windowsToWslPath(runnerPath),
    windowsToWslPath(pidPath),
    ...(syncOutput
      ? ["--sync-output", syncOutput.source, syncOutput.target]
      : []),
    ...command,
  ];
}

export function wslStagingOutputRoot(jobId: string): string {
  const safeId = jobId.replaceAll(/[^a-zA-Z0-9_.-]/g, "-");
  return `/tmp/mask-pipeline-studio/${safeId}/output`;
}

export async function terminateWslJob(
  settings: AppSettings,
  windowsPidPath: string,
): Promise<void> {
  if (settings.backendMode !== "wsl" || process.platform !== "win32") {
    return;
  }
  let pid: number;
  try {
    pid = Number.parseInt(fs.readFileSync(windowsPidPath, "utf8").trim(), 10);
  } catch {
    return;
  }
  if (!Number.isSafeInteger(pid) || pid <= 1) {
    return;
  }
  await new Promise<void>((resolve) => {
    const child = spawn(
      "wsl.exe",
      [
        ...distroArgs(settings),
        "--",
        "/bin/kill",
        "-TERM",
        "--",
        `-${pid}`,
      ],
      { windowsHide: true, stdio: "ignore" },
    );
    const done = () => resolve();
    child.once("error", done);
    child.once("close", done);
  });
}

/** Best-effort synchronous termination used only while Electron is quitting. */
export function terminateWslJobSync(
  settings: AppSettings,
  windowsPidPath: string,
): void {
  if (settings.backendMode !== "wsl" || process.platform !== "win32") {
    return;
  }
  let pid: number;
  try {
    pid = Number.parseInt(fs.readFileSync(windowsPidPath, "utf8").trim(), 10);
  } catch {
    return;
  }
  if (!Number.isSafeInteger(pid) || pid <= 1) {
    return;
  }
  spawnSync(
    "wsl.exe",
    [
      ...distroArgs(settings),
      "--",
      "/bin/kill",
      "-TERM",
      "--",
      `-${pid}`,
    ],
    { windowsHide: true, stdio: "ignore", timeout: 10_000 },
  );
}

export function runtimePathToHost(
  value: string,
  settings: AppSettings,
): string {
  if (
    settings.backendMode === "wsl" &&
    process.platform === "win32" &&
    value.startsWith("/")
  ) {
    return wslToWindowsPath(value, settings.wslDistro);
  }
  return value;
}
