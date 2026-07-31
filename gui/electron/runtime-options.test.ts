import { describe, expect, it } from "vitest";
import { parseRuntimeOptions } from "./runtime-options";

describe("Electron runtime options", () => {
  it("enables software rendering automatically in WSL", () => {
    expect(
      parseRuntimeOptions(["electron", "."], {
        WSL_DISTRO_NAME: "Ubuntu-24.04",
      }).softwareRendering,
    ).toBe(true);
  });

  it("allows hardware rendering to be requested explicitly", () => {
    expect(
      parseRuntimeOptions(["electron", ".", "--hardware-rendering"], {
        WSL_DISTRO_NAME: "Ubuntu-24.04",
      }).softwareRendering,
    ).toBe(false);
  });

  it("parses loopback automation settings without exposing them by default", () => {
    expect(
      parseRuntimeOptions(
        [
          "electron",
          ".",
          "--automation-port=9222",
          "--automation-video",
          "./a.mp4",
          "--automation-video=./b.mp4",
          "--automation-output=./runs",
        ],
        {},
      ),
    ).toMatchObject({
      automationPort: 9222,
      automationVideos: [
        expect.stringMatching(/a\.mp4$/),
        expect.stringMatching(/b\.mp4$/),
      ],
      automationOutput: expect.stringMatching(/runs$/),
      softwareRendering: false,
    });
  });

  it("rejects invalid automation ports", () => {
    expect(() =>
      parseRuntimeOptions(["electron", ".", "--automation-port=0"], {}),
    ).toThrow("automation-port");
  });
});
