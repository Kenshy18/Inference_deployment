import { describe, expect, it } from "vitest";
import { defaultDraft } from "./defaults";
import {
  isVideoPath,
  joinPath,
  settingsSummary,
  titleFromPath,
  uniqueOutputDir,
} from "./queue";

describe("titleFromPath", () => {
  it("strips directories and the extension", () => {
    expect(titleFromPath("/home/user/videos/scene-01.mp4")).toBe("scene-01");
    expect(titleFromPath("C:\\Users\\a\\clip.final.MOV")).toBe("clip.final");
    expect(titleFromPath("noext")).toBe("noext");
  });
});

describe("isVideoPath", () => {
  it("accepts the supported containers only", () => {
    expect(isVideoPath("/a/b.mp4")).toBe(true);
    expect(isVideoPath("/a/b.MKV")).toBe(true);
    expect(isVideoPath("/a/b.sqlite")).toBe(false);
  });
});

describe("joinPath", () => {
  it("keeps the separator style of the repository root", () => {
    expect(joinPath("/data/out/", "clip")).toBe("/data/out/clip");
    expect(joinPath("C:\\out", "clip")).toBe("C:\\out\\clip");
  });
});

describe("uniqueOutputDir", () => {
  it("uses the title and dedupes against claimed folders", () => {
    const first = uniqueOutputDir("/out", "clip", []);
    expect(first).toBe("/out/clip");
    expect(uniqueOutputDir("/out", "clip", [first, null])).toBe("/out/clip-2");
    expect(uniqueOutputDir("/out", "clip", [first, "/out/clip-2"])).toBe(
      "/out/clip-3",
    );
  });

  it("sanitizes characters that cannot appear in folder names", () => {
    expect(uniqueOutputDir("/out", 'a:b*c?"d', [])).toBe("/out/a_b_c__d");
  });
});

describe("settingsSummary", () => {
  it("describes the default draft", () => {
    const summary = settingsSummary(defaultDraft);
    expect(summary).toContain("Co-DINO（高速）");
    expect(summary).toContain("Face DINO v2（新）");
    expect(summary).toContain("楕円");
    expect(summary).toContain("overlay fast");
  });

  it("reflects reuse and disabled overlay", () => {
    const summary = settingsSummary({
      ...defaultDraft,
      inference: { ...defaultDraft.inference, enabled: false },
      overlay: { ...defaultDraft.overlay, enabled: false },
    });
    expect(summary).toContain("既存SQLite");
    expect(summary).toContain("overlayなし");
  });
});
