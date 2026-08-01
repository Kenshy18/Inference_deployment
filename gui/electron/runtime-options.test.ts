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
      automationAddress: "127.0.0.1",
      automationVideos: [
        expect.stringMatching(/a\.mp4$/),
        expect.stringMatching(/b\.mp4$/),
      ],
      automationOutput: expect.stringMatching(/runs$/),
      softwareRendering: false,
      qaE2eInput: null,
      qaE2eOutput: null,
      qaE2eReport: null,
      qaE2eMaxFrames: 120,
      qaE2eCancelAfterMs: null,
    });
  });

  it("parses packaged Windows end-to-end QA inputs", () => {
    expect(
      parseRuntimeOptions(
        [
          "electron",
          ".",
          "--qa-e2e-input=C:\\video\\input.mp4",
          "--qa-e2e-output=D:\\qa output",
          "--qa-e2e-report=D:\\qa output\\report.json",
          "--qa-e2e-max-frames=240",
          "--qa-e2e-cancel-after-ms=2500",
        ],
        {},
      ),
    ).toMatchObject({
      qaE2eInput: expect.stringMatching(/input\.mp4$/),
      qaE2eOutput: expect.stringMatching(/qa output$/),
      qaE2eReport: expect.stringMatching(/report\.json$/),
      qaE2eMaxFrames: 240,
      qaE2eCancelAfterMs: 2500,
    });
  });

  it("rejects an unrealistically short cancellation delay", () => {
    expect(() =>
      parseRuntimeOptions(
        ["electron", ".", "--qa-e2e-cancel-after-ms=50"],
        {},
      ),
    ).toThrow("qa-e2e-cancel-after-ms");
  });

  it("rejects invalid automation ports", () => {
    expect(() =>
      parseRuntimeOptions(["electron", ".", "--automation-port=0"], {}),
    ).toThrow("automation-port");
  });

  it("only exposes automation outside loopback when explicitly requested", () => {
    expect(() =>
      parseRuntimeOptions(
        ["electron", ".", "--automation-port=9222"],
        { MASK_STUDIO_AUTOMATION_ADDRESS: "0.0.0.0" },
      ),
    ).toThrow("MASK_STUDIO_ALLOW_REMOTE_AUTOMATION");
    expect(
      parseRuntimeOptions(
        ["electron", ".", "--automation-port=9222"],
        {
          MASK_STUDIO_AUTOMATION_ADDRESS: "0.0.0.0",
          MASK_STUDIO_ALLOW_REMOTE_AUTOMATION: "1",
        },
      ).automationAddress,
    ).toBe("0.0.0.0");
    expect(() =>
      parseRuntimeOptions(
        ["electron", ".", "--automation-port=9222"],
        { MASK_STUDIO_AUTOMATION_ADDRESS: "192.168.0.1" },
      ),
    ).toThrow("automation-address");
  });
});
