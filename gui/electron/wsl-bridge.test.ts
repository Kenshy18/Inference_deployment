import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { AppSettings } from "../shared/types";
import {
  isWslPathInside,
  ensureWslDriveMounts,
  validateWslJobPaths,
  WSL_RUNNER_SOURCE,
  windowsToWslPath,
  windowsDriveLetters,
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

  it("collects distinct drive-letter paths recursively", () => {
    expect(
      windowsDriveLetters({
        input: "O:\\video\\source.mp4",
        output: "D:/results",
        nested: ["o:/models/file.engine", "/home/kenshin/local"],
      }),
    ).toEqual(["D", "O"]);
  });

  it("mounts only missing drives using argv-safe commands", async () => {
    const calls: Array<{ args: string[]; user?: string }> = [];
    const runner = async (
      _settings: AppSettings,
      args: string[],
      _cwd?: string,
      user?: string,
    ): Promise<{ stdout: string; stderr: string }> => {
      calls.push({ args, user });
      if (args[0] === "/usr/bin/mountpoint" && args[2] === "/mnt/o") {
        const oChecks = calls.filter(
          (call) =>
            call.args[0] === "/usr/bin/mountpoint" &&
            call.args[2] === "/mnt/o",
        ).length;
        if (oChecks === 1) {
          throw new Error("not mounted");
        }
      }
      return { stdout: "", stderr: "" };
    };
    await expect(
      ensureWslDriveMounts(
        settings,
        ["C:\\local\\input.mp4", "O:\\network\\input.mp4"],
        "win32",
        runner,
      ),
    ).resolves.toEqual(["O"]);
    expect(calls).toContainEqual({
      args: ["/usr/bin/mkdir", "-p", "/mnt/o"],
      user: "root",
    });
    expect(calls).toContainEqual({
      args: ["/usr/bin/mount", "-t", "drvfs", "O:", "/mnt/o"],
      user: "root",
    });
    expect(calls.every((call) => !call.args.includes("/bin/sh"))).toBe(true);
  });

  it("rejects raw network UNC paths with an actionable message", async () => {
    await expect(
      ensureWslDriveMounts(
        settings,
        { input: "\\\\server\\share\\movie.mp4" },
        "win32",
        async () => ({ stdout: "", stderr: "" }),
      ),
    ).rejects.toThrow("ドライブ文字へ割り当て");
  });

  it.skipIf(process.platform === "win32")(
    "executes the generated WSL path-validation program",
    async () => {
      const root = fs.mkdtempSync(
        path.join(os.tmpdir(), "mask-studio-wsl-path-validation-"),
      );
      const input = path.join(root, "input.mp4");
      const output = path.join(root, "new", "nested", "output");
      fs.writeFileSync(input, "fixture\n", "utf8");
      try {
        await expect(
          validateWslJobPaths(
            settings,
            input,
            output,
            "win32",
            async (_settings, args) => {
              const result = spawnSync(args[0], args.slice(1), {
                encoding: "utf8",
              });
              if (result.status !== 0) {
                throw new Error(result.stderr || result.stdout);
              }
              return { stdout: result.stdout, stderr: result.stderr };
            },
          ),
        ).resolves.toBeUndefined();
      } finally {
        fs.rmSync(root, { recursive: true, force: true });
      }
    },
  );

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
      expect(fs.existsSync(path.dirname(source))).toBe(false);
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

  it.skipIf(process.platform === "win32")(
    "copies only manifest-published artifacts from orchestration staging",
    () => {
      const root = fs.mkdtempSync(
        path.join(os.tmpdir(), "mask-studio-wsl-runner-compact-test-"),
      );
      const source = path.join(
        "/tmp/mask-pipeline-studio",
        `vitest-compact-${process.pid}-${Date.now()}`,
        "output",
      );
      const target = path.join(root, "windows-output");
      const pidFile = path.join(root, "runner.pid");
      const resultSqlite = path.join(source, "sample.sqlite");
      const overlay = path.join(source, "overlay", "combined_simple.mp4");
      const log = path.join(source, "logs", "postprocess.log");
      const intermediate = path.join(
        source,
        "logs",
        "work",
        "02_postprocess",
        "04_classwise_postprocess",
        "reproducible.sqlite",
      );
      fs.mkdirSync(path.dirname(resultSqlite), { recursive: true });
      fs.mkdirSync(path.dirname(overlay), { recursive: true });
      fs.mkdirSync(path.dirname(log), { recursive: true });
      fs.mkdirSync(path.dirname(intermediate), { recursive: true });
      fs.writeFileSync(resultSqlite, "sqlite\n", "utf8");
      fs.writeFileSync(overlay, "video\n", "utf8");
      fs.writeFileSync(log, "log\n", "utf8");
      fs.writeFileSync(intermediate, "large temporary data\n", "utf8");
      fs.writeFileSync(
        path.join(source, "logs", "resolved_config.json"),
        `${JSON.stringify({ output_root: source })}\n`,
        "utf8",
      );
      fs.writeFileSync(
        path.join(source, "logs", "run_manifest.json"),
        `${JSON.stringify({
          status: "complete",
          output_root: source,
          artifacts: {
            result_sqlite: resultSqlite,
            overlay_combined_simple: overlay,
          },
        })}\n`,
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
        expect(fs.readFileSync(path.join(target, "logs", "postprocess.log"), "utf8")).toBe(
          "log\n",
        );
        expect(
          fs.readFileSync(path.join(target, "sample.sqlite"), "utf8"),
        ).toBe("sqlite\n");
        expect(
          fs.readFileSync(
            path.join(target, "overlay", "combined_simple.mp4"),
            "utf8",
          ),
        ).toBe("video\n");
        expect(
          fs.existsSync(
            path.join(
              target,
              "logs",
              "work",
              "02_postprocess",
              "04_classwise_postprocess",
              "reproducible.sqlite",
            ),
          ),
        ).toBe(false);
        const manifest = JSON.parse(
          fs.readFileSync(
            path.join(target, "logs", "run_manifest.json"),
            "utf8",
          ),
        ) as { artifacts: { result_sqlite: string } };
        expect(manifest.artifacts.result_sqlite).toBe(
          path.join(target, "sample.sqlite"),
        );
        expect(fs.existsSync(source)).toBe(false);
      } finally {
        fs.rmSync(root, { recursive: true, force: true });
        fs.rmSync(path.dirname(source), { recursive: true, force: true });
      }
    },
  );

  it.skipIf(process.platform === "win32")(
    "removes a dead marked staging job before starting a new job",
    () => {
      const root = fs.mkdtempSync(
        path.join(os.tmpdir(), "mask-studio-wsl-runner-stale-test-"),
      );
      const stagingRoot = "/tmp/mask-pipeline-studio";
      const stale = path.join(
        stagingRoot,
        ["vitest-stale", process.pid, Date.now()].join("-"),
      );
      const source = path.join(
        stagingRoot,
        ["vitest-current", process.pid, Date.now()].join("-"),
        "output",
      );
      const target = path.join(root, "windows-output");
      const pidFile = path.join(root, "runner.pid");
      fs.mkdirSync(stale, { recursive: true });
      fs.writeFileSync(
        path.join(stale, ".wsl-runner.json"),
        JSON.stringify({
          schema_version: 1,
          runner: { pid: 999_999_999, start_ticks: "0" },
          child: { pid: 999_999_998, start_ticks: "0" },
        }) + "\n",
        "utf8",
      );
      fs.writeFileSync(path.join(stale, "large.partial"), "stale\n", "utf8");
      fs.mkdirSync(source, { recursive: true });
      fs.writeFileSync(path.join(source, "result.txt"), "ok\n", "utf8");
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
        expect(fs.existsSync(stale)).toBe(false);
        expect(fs.readFileSync(path.join(target, "result.txt"), "utf8")).toBe(
          "ok\n",
        );
      } finally {
        fs.rmSync(root, { recursive: true, force: true });
        fs.rmSync(stale, { recursive: true, force: true });
        fs.rmSync(path.dirname(source), { recursive: true, force: true });
      }
    },
  );

  it.skipIf(process.platform === "win32")(
    "preserves staging owned by a live process",
    () => {
      const root = fs.mkdtempSync(
        path.join(os.tmpdir(), "mask-studio-wsl-runner-live-test-"),
      );
      const stagingRoot = "/tmp/mask-pipeline-studio";
      const active = path.join(
        stagingRoot,
        ["vitest-active", process.pid, Date.now()].join("-"),
      );
      const source = path.join(
        stagingRoot,
        ["vitest-current", process.pid, Date.now()].join("-"),
        "output",
      );
      const target = path.join(root, "windows-output");
      const pidFile = path.join(root, "runner.pid");
      const statSuffix = fs
        .readFileSync(path.join("/proc", String(process.pid), "stat"), "ascii")
        .split(/\)\s/, 2)[1]
        .trim()
        .split(/\s+/);
      const startTicks = statSuffix[19];
      fs.mkdirSync(active, { recursive: true });
      fs.writeFileSync(
        path.join(active, ".wsl-runner.json"),
        JSON.stringify({
          schema_version: 1,
          runner: { pid: process.pid, start_ticks: startTicks },
        }) + "\n",
        "utf8",
      );
      fs.mkdirSync(source, { recursive: true });
      fs.writeFileSync(path.join(source, "result.txt"), "ok\n", "utf8");
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
        expect(fs.existsSync(active)).toBe(true);
      } finally {
        fs.rmSync(root, { recursive: true, force: true });
        fs.rmSync(active, { recursive: true, force: true });
        fs.rmSync(path.dirname(source), { recursive: true, force: true });
      }
    },
  );
});
