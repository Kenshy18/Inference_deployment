import { EventEmitter } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ChildProcess } from "node:child_process";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppSettings, PipelineDraft } from "../shared/types";
import { defaultDraft } from "../src/lib/defaults";

const mocks = vi.hoisted(() => ({
  spawn: vi.fn(),
  validate: vi.fn(),
}));

vi.mock("node:child_process", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:child_process")>()),
  spawn: mocks.spawn,
}));

vi.mock("./wsl-bridge", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./wsl-bridge")>()),
  validateWslBackend: mocks.validate,
}));

import { JobManager, pruneJobDirectories } from "./job-manager";

class FakeChild extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  pid = 1234;
  kill = vi.fn(() => true);
}

const settings: AppSettings = {
  backendMode: "native",
  backendRoot: "/opt/inference_backend2",
  runtimePython: "/opt/runtime/bin/python",
  wslDistro: "Ubuntu-24.04",
};

function draft(root: string, name: string): PipelineDraft {
  return {
    ...structuredClone(defaultDraft),
    inputVideo: path.join(root, `${name}.mp4`),
    outputRoot: path.join(root, name),
  };
}

describe("JobManager run ownership", () => {
  let temporary: string;
  let children: FakeChild[];

  beforeEach(() => {
    temporary = fs.mkdtempSync(path.join(os.tmpdir(), "job-manager-test-"));
    children = [];
    mocks.spawn.mockReset();
    mocks.spawn.mockImplementation(() => {
      const child = new FakeChild();
      children.push(child);
      return child as unknown as ChildProcess;
    });
    mocks.validate.mockReset();
    mocks.validate.mockResolvedValue(undefined);
  });

  afterEach(() => fs.rmSync(temporary, { recursive: true, force: true }));

  it("reserves ownership while asynchronous backend validation is pending", async () => {
    let releaseValidation: (() => void) | undefined;
    mocks.validate.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          releaseValidation = resolve;
        }),
    );
    const manager = new JobManager(path.join(temporary, "jobs"));
    const first = manager.run(draft(temporary, "dry"), settings, true);
    await vi.waitFor(() => expect(releaseValidation).toBeTypeOf("function"));

    await expect(
      manager.run(draft(temporary, "real"), settings, false),
    ).rejects.toThrow("別のジョブが実行中です");

    releaseValidation?.();
    await first;
    expect(children).toHaveLength(1);
    children[0].emit("close", 0);
    expect(manager.snapshot().status).toBe("validated");
  });

  it("does not let callbacks from a completed Dry Run mutate the next run", async () => {
    const manager = new JobManager(path.join(temporary, "jobs"));
    await manager.run(draft(temporary, "dry"), settings, true);
    const dryChild = children[0];
    dryChild.emit("close", 0);
    expect(manager.snapshot().status).toBe("validated");

    await manager.run(draft(temporary, "real"), settings, false);
    const realId = manager.snapshot().id;
    expect(manager.snapshot().status).toBe("running");

    dryChild.emit("error", new Error("late Dry Run callback"));
    expect(manager.snapshot().id).toBe(realId);
    expect(manager.snapshot().status).toBe("running");

    children[1].emit("close", 0);
    expect(manager.snapshot().id).toBe(realId);
    expect(manager.snapshot().status).toBe("completed");
  });
});

describe("JobManager metadata retention", () => {
  it("removes expired and excess job directories but preserves unrelated data", () => {
    const temporary = fs.mkdtempSync(
      path.join(os.tmpdir(), "job-manager-retention-test-"),
    );
    const now = Date.UTC(2026, 7, 3, 0, 0, 0);
    const names = [
      "2026-08-02T00-00-00-000Z",
      "2026-08-01T00-00-00-000Z",
      "2026-07-31T00-00-00-000Z",
      "2026-06-01T00-00-00-000Z",
    ];
    try {
      for (const [index, name] of names.entries()) {
        const directory = path.join(temporary, name);
        fs.mkdirSync(directory);
        fs.writeFileSync(path.join(directory, "job.json"), "{}\n", "utf8");
        const timestamp = new Date(
          now - (index + 1) * 24 * 60 * 60 * 1_000,
        );
        fs.utimesSync(directory, timestamp, timestamp);
      }
      fs.mkdirSync(path.join(temporary, "user-data"));

      pruneJobDirectories(temporary, now, 30, 2);

      expect(fs.existsSync(path.join(temporary, names[0]))).toBe(true);
      expect(fs.existsSync(path.join(temporary, names[1]))).toBe(true);
      expect(fs.existsSync(path.join(temporary, names[2]))).toBe(false);
      expect(fs.existsSync(path.join(temporary, names[3]))).toBe(false);
      expect(fs.existsSync(path.join(temporary, "user-data"))).toBe(true);
    } finally {
      fs.rmSync(temporary, { recursive: true, force: true });
    }
  });
});
