import { execFile } from "node:child_process";
import os from "node:os";
import { promisify } from "node:util";
import type { HardwareMetrics } from "../shared/types";

const execFileAsync = promisify(execFile);

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

async function readNvidiaMetrics(): Promise<NvidiaMetrics | null> {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      [
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
      ],
      { timeout: 1_500, maxBuffer: 64 * 1_024 },
    );
    return parseNvidiaSmi(stdout);
  } catch {
    // The GUI also runs on CPU-only machines. Missing GPU telemetry must not
    // affect workflow execution or turn into repeated console noise.
    return null;
  }
}

export class HardwareSampler {
  private previousCpu: CpuTicks | null = null;

  async sample(): Promise<HardwareMetrics> {
    const currentCpu = readCpuTicks();
    const cpuPercent = calculateCpuPercent(this.previousCpu, currentCpu);
    this.previousCpu = currentCpu;

    const memoryTotalBytes = os.totalmem();
    const memoryUsedBytes = Math.max(0, memoryTotalBytes - os.freemem());
    const gpu = await readNvidiaMetrics();

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
}
