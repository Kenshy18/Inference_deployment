import { execFile, spawn, type ChildProcess } from "node:child_process";
import os from "node:os";
import type { HardwareMetrics } from "../shared/types";

export interface CpuTicks {
  idle: number;
  total: number;
}

export interface NvidiaMetrics {
  gpuPercent: number;
  vramPercent: number;
  vramUsedMiB: number;
  vramTotalMiB: number;
  gpuTemperatureC: number;
}

export function readCpuTicks(): CpuTicks {
  return os.cpus().reduce<CpuTicks>(
    (sum, cpu) => {
      const times = Object.values(cpu.times);
      return {
        idle: sum.idle + cpu.times.idle,
        total: sum.total + times.reduce((total, value) => total + value, 0),
      };
    },
    { idle: 0, total: 0 },
  );
}

export function calculateCpuPercent(
  previous: CpuTicks | null,
  current: CpuTicks,
): number | null {
  if (!previous) {
    return null;
  }
  const totalDelta = current.total - previous.total;
  const idleDelta = current.idle - previous.idle;
  if (totalDelta <= 0) {
    return null;
  }
  return Math.min(100, Math.max(0, (1 - idleDelta / totalDelta) * 100));
}

export function parseNvidiaSmi(stdout: string): NvidiaMetrics | null {
  const firstLine = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
  if (!firstLine) {
    return null;
  }
  const values = firstLine.split(",").map((value) => Number(value.trim()));
  if (values.length < 4 || values.some((value) => !Number.isFinite(value))) {
    return null;
  }
  const [gpuPercent, vramUsedMiB, vramTotalMiB, gpuTemperatureC] = values;
  if (vramTotalMiB <= 0) {
    return null;
  }
  return {
    gpuPercent: Math.min(100, Math.max(0, gpuPercent)),
    vramPercent: Math.min(100, Math.max(0, (vramUsedMiB / vramTotalMiB) * 100)),
    vramUsedMiB,
    vramTotalMiB,
    gpuTemperatureC,
  };
}

export function parseNvidiaDmonLine(line: string): NvidiaMetrics | null {
  const values = line.trim().split(/\s+/);
  if (values.length < 16 || values[0] === "#") {
    return null;
  }
  // dmon -s pucvmt columns: gpu, pwr, gtemp, mtemp, sm, mem, enc,
  // dec, jpg, ofa, mclk, pclk, pviol, tviol, fb, bar1, ...
  const gpuPercent = Number(values[4]);
  const gpuTemperatureC = Number(values[2]);
  const vramUsedMiB = Number(values[14]);
  const vramTotalMiB = Number(values[15]);
  if (
    ![gpuPercent, gpuTemperatureC, vramUsedMiB, vramTotalMiB].every(
      Number.isFinite,
    ) ||
    vramTotalMiB <= 0
  ) {
    return null;
  }
  return {
    gpuPercent: Math.min(100, Math.max(0, gpuPercent)),
    vramPercent: Math.min(100, Math.max(0, (vramUsedMiB / vramTotalMiB) * 100)),
    vramUsedMiB,
    vramTotalMiB,
    gpuTemperatureC,
  };
}

export class HardwareSampler {
  private previousCpu: CpuTicks | null = null;
  private gpu: NvidiaMetrics | null = null;
  private dmon: ChildProcess | null = null;
  private dmonBuffer = "";
  private retryDmonAfter = 0;
  private dmonHasSamples = false;
  private fallbackPending = false;
  private nextFallbackAt = 0;

  private ensureDmon(): void {
    if (this.dmon !== null || Date.now() < this.retryDmonAfter) {
      return;
    }
    const child = spawn("nvidia-smi", ["dmon", "-s", "pucvmt", "-d", "1"], {
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
    });
    this.dmon = child;
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      const lines = `${this.dmonBuffer}${chunk}`.split(/\r?\n/);
      this.dmonBuffer = lines.pop() ?? "";
      for (const line of lines) {
        const parsed = parseNvidiaDmonLine(line);
        if (parsed !== null) {
          this.gpu = parsed;
          this.dmonHasSamples = true;
        }
      }
    });
    const stopped = () => {
      if (this.dmon === child) {
        this.dmon = null;
        this.dmonBuffer = "";
        this.dmonHasSamples = false;
        this.retryDmonAfter = Date.now() + 10_000;
      }
    };
    child.once("error", stopped);
    child.once("close", stopped);
  }

  private ensureFallbackSample(): void {
    if (
      this.dmonHasSamples ||
      this.fallbackPending ||
      Date.now() < this.nextFallbackAt
    ) {
      return;
    }
    this.fallbackPending = true;
    this.nextFallbackAt = Date.now() + 1_000;
    execFile(
      "nvidia-smi",
      [
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
      ],
      { timeout: 3_000, windowsHide: true },
      (_error, stdout) => {
        this.fallbackPending = false;
        const parsed = parseNvidiaSmi(stdout);
        if (parsed !== null) {
          this.gpu = parsed;
        }
      },
    );
  }

  async sample(): Promise<HardwareMetrics> {
    this.ensureDmon();
    this.ensureFallbackSample();
    const currentCpu = readCpuTicks();
    const cpuPercent = calculateCpuPercent(this.previousCpu, currentCpu);
    this.previousCpu = currentCpu;

    const memoryTotalBytes = os.totalmem();
    const memoryUsedBytes = Math.max(0, memoryTotalBytes - os.freemem());
    const gpu = this.gpu;

    return {
      timestamp: Date.now(),
      cpuPercent,
      memoryPercent:
        memoryTotalBytes > 0 ? (memoryUsedBytes / memoryTotalBytes) * 100 : null,
      memoryUsedBytes,
      memoryTotalBytes,
      gpuPercent: gpu?.gpuPercent ?? null,
      vramPercent: gpu?.vramPercent ?? null,
      vramUsedMiB: gpu?.vramUsedMiB ?? null,
      vramTotalMiB: gpu?.vramTotalMiB ?? null,
      gpuTemperatureC: gpu?.gpuTemperatureC ?? null,
    };
  }

  close(): void {
    const child = this.dmon;
    this.dmon = null;
    if (child !== null && child.exitCode === null) {
      child.kill("SIGTERM");
    }
  }
}
