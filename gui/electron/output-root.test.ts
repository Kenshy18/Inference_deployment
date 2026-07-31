import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { availableOutputRoot } from "./output-root";

const temporaryRoots: string[] = [];

function temporary(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mask-output-root-"));
  temporaryRoots.push(root);
  return root;
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

describe("availableOutputRoot", () => {
  it("keeps a missing or empty requested directory", () => {
    const root = temporary();
    expect(availableOutputRoot(path.join(root, "missing"), false)).toBe(
      path.join(root, "missing"),
    );
    const empty = path.join(root, "empty");
    fs.mkdirSync(empty);
    expect(availableOutputRoot(empty, false)).toBe(empty);
  });

  it("increments past existing results and partial output", () => {
    const root = temporary();
    const requested = path.join(root, "video");
    fs.mkdirSync(requested);
    fs.writeFileSync(path.join(requested, "run_manifest.json"), "{}");
    fs.mkdirSync(`${requested}_2`);
    fs.writeFileSync(path.join(`${requested}_2`, "partial.log"), "partial");
    expect(availableOutputRoot(requested, false)).toBe(`${requested}_3`);
  });

  it("keeps the requested directory for explicit resume", () => {
    const root = temporary();
    const requested = path.join(root, "video");
    fs.mkdirSync(requested);
    fs.writeFileSync(path.join(requested, "run_manifest.json"), "{}");
    expect(availableOutputRoot(requested, true)).toBe(requested);
  });
});
