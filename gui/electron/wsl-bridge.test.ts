import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { AppSettings } from "../shared/types";
import {
  isWslPathInside,
  WSL_RUNNER_SOURCE,
  windowsToWslPath,
  wslLaunchWrapper,
  wslToWindowsPath,
} from "./wsl-bridge";

const settings: AppSettings = {
  backendMode: "wsl",
  backendRoot: "/home/kenshin/inference_backend2",
  runtimePython:
    "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
  wslDistro: "Ubuntu-24.04",
};

describe("Windows/WSL bridge", () => {
  it("maps drive and UNC paths into WSL", () => {
    expect(windowsToWslPath("D:\\jobs\\movie.mp4")).toBe(
      "/mnt/d/jobs/movie.mp4",
    );
    expect(
      windowsToWslPath(
        "\\\\wsl.localhost\\Ubuntu-24.04\\home\\kenshin\\movie.mp4",
      ),
    ).toBe("/home/kenshin/movie.mp4");
  });

  it("maps WSL paths back to Windows without a subprocess", () => {
    expect(wslToWindowsPath("/mnt/d/jobs/out.mp4", "Ubuntu-24.04")).toBe(
      "D:\\jobs\\out.mp4",
    );
    expect(wslToWindowsPath("/home/kenshin/out.mp4", "Ubuntu-24.04")).toBe(
      "\\\\wsl.localhost\\Ubuntu-24.04\\home\\kenshin\\out.mp4",
    );
  });

  it("rejects prefix-confusion paths", () => {
    expect(isWslPathInside("/mnt/d/out/live.jpg", "/mnt/d/out")).toBe(true);
    expect(isWslPathInside("/mnt/d/output/live.jpg", "/mnt/d/out")).toBe(
      false,
    );
  });

  it("wraps the workflow in a cancellable Linux process group", () => {
    const args = wslLaunchWrapper(
      settings,
      "D:\\jobs\\run.pid",
      "D:\\jobs\\wsl-runner.py",
      [
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        settings.runtimePython,
        "-m",
        "orchestration",
      ],
    );
    expect(args.slice(0, 4)).toEqual([
      "-d",
      "Ubuntu-24.04",
      "--cd",
      "/home/kenshin/inference_backend2",
    ]);
    expect(args).toContain("/mnt/d/jobs/run.pid");
    expect(args).toContain("PYTHONUNBUFFERED=1");
    expect(args).toContain("/mnt/d/jobs/wsl-runner.py");
    expect(args).not.toContain("/bin/sh");
  });

  it("passes an ext4-to-Windows output sync without shell parsing", () => {
    const args = wslLaunchWrapper(
      settings,
      "D:\\jobs\\run.pid",
      "D:\\jobs\\wsl-runner.py",
      [settings.runtimePython, "-m", "orchestration"],
      {
        source: "/tmp/mask-pipeline-studio/job/output",
        target: "/mnt/d/新しいフォルダー/output",
      },
    );
    expect(args).toContain("--sync-output");
    expect(args).toContain("/mnt/d/新しいフォルダー/output");
  });

  it.skipIf(process.platform === "win32")(
    "copies a successful staged output atomically and removes staging",
    () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "mask-studio-wsl-runner-test-"),
    );
    const source = path.join(
      "/tmp/mask-pipeline-studio",
      `vitest-${process.pid}-${Date.now()}`,
      "output",
    );
    const target = path.join(root, "windows-output");
    const pidFile = path.join(root, "runner.pid");
    fs.mkdirSync(source, { recursive: true });
    fs.writeFileSync(path.join(source, "result.txt"), "ok\n", "utf8");
    fs.writeFileSync(
      path.join(source, "manifest.json"),
      `${JSON.stringify({ output_root: source })}\n`,
      "utf8",
    );
    try {
      const bootstrap =
        "import sys; source=sys.stdin.read(); " +
        "exec(compile(source, '<wsl-runner>', 'exec'), {'__name__':'__main__'})";
      const result = spawnSync(
        "python3",
        [
          "-c",
          bootstrap,
          pidFile,
          "--sync-output",
          source,
          target,
          "/bin/true",
        ],
        { input: WSL_RUNNER_SOURCE, encoding: "utf8" },
      );
      expect(result.status, result.stderr).toBe(0);
      expect(fs.readFileSync(path.join(target, "result.txt"), "utf8")).toBe(
        "ok\n",
      );
      const manifest = JSON.parse(
        fs.readFileSync(path.join(target, "manifest.json"), "utf8"),
      ) as { output_root: string };
      expect(manifest.output_root).toBe(target);
      expect(fs.existsSync(source)).toBe(false);
      expect(fs.existsSync(`${target}.partial`)).toBe(false);
      expect(fs.existsSync(pidFile)).toBe(false);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
      fs.rmSync(path.dirname(source), {
        recursive: true,
        force: true,
      });
    }
    },
  );
});
