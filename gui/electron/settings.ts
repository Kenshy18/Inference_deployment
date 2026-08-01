import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AppSettings } from "../shared/types";

function defaultBackendRoot(): string {
  const explicit = process.env.INFERENCE_BACKEND_ROOT;
  if (explicit) {
    return explicit;
  }
  if (process.platform === "win32") {
    return "/home/kenshin/inference_backend2";
  }
  const sibling = path.join(os.homedir(), "inference_backend2");
  return sibling;
}

function defaultRuntimePython(): string {
  const explicit = process.env.INFERENCE_RUNTIME_PYTHON;
  if (explicit) {
    return explicit;
  }
  if (process.platform === "win32") {
    return "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10";
  }
  const production = path.join(
    os.homedir(),
    ".local",
    "share",
    "video-mask-runtime",
    "envs",
    "production",
    "bin",
    "python3.10",
  );
  return fs.existsSync(production) ? production : "python3";
}

export function defaultSettings(): AppSettings {
  return {
    backendMode: process.platform === "win32" ? "wsl" : "native",
    backendRoot: defaultBackendRoot(),
    runtimePython: defaultRuntimePython(),
    wslDistro: "Ubuntu-24.04",
  };
}

function normalizeSettings(value: Partial<AppSettings>): AppSettings {
  const defaults = defaultSettings();
  return {
    backendMode:
      process.platform === "win32"
        ? "wsl"
        : value.backendMode === "wsl" || value.backendMode === "native"
        ? value.backendMode
        : defaults.backendMode,
    backendRoot:
      typeof value.backendRoot === "string"
        ? value.backendRoot
        : defaults.backendRoot,
    runtimePython:
      typeof value.runtimePython === "string"
        ? value.runtimePython
        : defaults.runtimePython,
    wslDistro:
      typeof value.wslDistro === "string"
        ? value.wslDistro
        : defaults.wslDistro,
  };
}

export function readSettings(settingsPath: string): AppSettings {
  try {
    const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as object;
    return normalizeSettings(raw);
  } catch {
    return defaultSettings();
  }
}

export function writeSettings(
  settingsPath: string,
  value: AppSettings,
): AppSettings {
  const settings = normalizeSettings(value);
  fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
  const temporary = `${settingsPath}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(settings, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, settingsPath);
  return settings;
}
