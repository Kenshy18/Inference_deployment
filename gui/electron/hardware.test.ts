import { describe, expect, it } from "vitest";
import { calculateCpuPercent, parseNvidiaSmi } from "./hardware";

describe("hardware telemetry", () => {
  it("calculates CPU utilization from cumulative ticks", () => {
    expect(
      calculateCpuPercent(
        { idle: 400, total: 1_000 },
        { idle: 450, total: 1_200 },
      ),
    ).toBe(75);
    expect(calculateCpuPercent(null, { idle: 0, total: 0 })).toBeNull();
  });

  it("parses the first NVIDIA GPU without locale-dependent labels", () => {
    expect(parseNvidiaSmi("71, 6144, 12288, 67\n4, 20, 100, 42\n")).toEqual({
      gpuPercent: 71,
      vramPercent: 50,
      vramUsedMiB: 6144,
      vramTotalMiB: 12288,
      gpuTemperatureC: 67,
    });
  });

  it("treats malformed GPU output as unavailable", () => {
    expect(parseNvidiaSmi("N/A, 100, 1000, 40")).toBeNull();
    expect(parseNvidiaSmi("")).toBeNull();
  });
});
