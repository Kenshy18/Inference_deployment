import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  bindDeploymentSettings,
  loadDeploymentProfile,
} from "./deployment-profile";

const temporaryRoots: string[] = [];

function temporaryRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mask-profile-"));
  temporaryRoots.push(root);
  return root;
}

function writeProfile(root: string, distro = "MaskPipelineProduction-20260818") {
  const userData = path.join(root, "user-data");
  fs.writeFileSync(
    path.join(root, "deployment-profile.json"),
    `${JSON.stringify({
      schema_version: 1,
      release_id: "mask-pipeline-20260818-c86cbbf",
      user_data_path: userData,
      backend_root: "/home/kenshin/inference_backend2",
      runtime_python:
        "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
      wsl_distribution: distro,
    })}\n`,
  );
  return userData;
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

describe("deployment profile", () => {
  it("discovers the profile beside a portable executable", () => {
    const root = temporaryRoot();
    const userData = writeProfile(root);
    const profile = loadDeploymentProfile(
      ["Mask Pipeline Studio.exe"],
      { PORTABLE_EXECUTABLE_DIR: root },
      path.join(root, "temporary", "Mask Pipeline Studio.exe"),
    );
    expect(profile).toMatchObject({
      releaseId: "mask-pipeline-20260818-c86cbbf",
      userDataPath: userData,
      wslDistro: "MaskPipelineProduction-20260818",
    });
  });

  it("keeps separately installed releases isolated", () => {
    const oldRoot = temporaryRoot();
    const newRoot = temporaryRoot();
    writeProfile(oldRoot, "MaskPipelineProduction-20260817");
    writeProfile(newRoot, "MaskPipelineProduction-20260818");
    const oldProfile = loadDeploymentProfile([], {
      PORTABLE_EXECUTABLE_DIR: oldRoot,
    }, path.join(oldRoot, "app.exe"));
    const newProfile = loadDeploymentProfile([], {
      PORTABLE_EXECUTABLE_DIR: newRoot,
    }, path.join(newRoot, "app.exe"));
    expect(oldProfile?.userDataPath).not.toBe(newProfile?.userDataPath);
    expect(oldProfile?.wslDistro).toBe("MaskPipelineProduction-20260817");
    expect(newProfile?.wslDistro).toBe("MaskPipelineProduction-20260818");
  });

  it("rejects a missing explicitly requested profile", () => {
    expect(() =>
      loadDeploymentProfile(
        ["app", "--deployment-profile=/missing/profile.json"],
        {},
        "/tmp/app",
      ),
    ).toThrow("does not exist");
  });

  it("rejects unsafe distribution identifiers", () => {
    const root = temporaryRoot();
    writeProfile(root, "../unsafe");
    expect(() =>
      loadDeploymentProfile([], { PORTABLE_EXECUTABLE_DIR: root }, "/tmp/app"),
    ).toThrow("unsafe");
  });

  it("pins saved runtime settings to the release backend", () => {
    const root = temporaryRoot();
    writeProfile(root);
    const profile = loadDeploymentProfile([], {
      PORTABLE_EXECUTABLE_DIR: root,
    }, "/tmp/app");
    expect(
      bindDeploymentSettings(
        {
          backendMode: "native",
          backendRoot: "/wrong/backend",
          runtimePython: "python3",
          wslDistro: "WrongDistribution",
        },
        profile,
      ),
    ).toMatchObject({
      backendMode: "wsl",
      backendRoot: "/home/kenshin/inference_backend2",
      runtimePython:
        "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
      wslDistro: "MaskPipelineProduction-20260818",
    });
  });
});
