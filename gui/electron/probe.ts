import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AppSettings, VideoProbe } from "../shared/types";

const EXEC_TIMEOUT_MS = 10_000;

function run(bin: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      bin,
      args,
      { timeout: EXEC_TIMEOUT_MS, maxBuffer: 8 * 1024 * 1024 },
      (error, stdout) => (error ? reject(error) : resolve(stdout)),
    );
  });
}

/** Bundled FFmpeg of the backend repo, then whatever is on PATH. */
function binCandidates(name: string, settings: AppSettings): string[] {
  const candidates: string[] = [];
  if (settings.backendMode === "native" && settings.backendRoot.trim()) {
    candidates.push(
      path.join(
        settings.backendRoot,
        "overlay",
        ".runtime",
        "ffmpeg-nvenc",
        "bin",
        name,
      ),
    );
  }
  candidates.push(name);
  return candidates;
}

async function firstWorking(
  candidates: string[],
  args: string[],
): Promise<string | null> {
  for (const bin of candidates) {
    try {
      await run(bin, args);
      return bin;
    } catch {
      /* try next */
    }
  }
  return null;
}

/** Duration + poster frame for a queue entry. Best-effort: every failure
 *  degrades to null so the queue still works without FFmpeg. */
export async function probeVideo(
  videoPath: string,
  settings: AppSettings,
): Promise<VideoProbe> {
  const result: VideoProbe = { durationSeconds: null, thumbnail: null };
  if (!videoPath.trim()) {
    return result;
  }

  const ffprobe = await firstWorking(binCandidates("ffprobe", settings), [
    "-version",
  ]);
  if (ffprobe) {
    try {
      const out = await run(ffprobe, [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        videoPath,
      ]);
      const seconds = Number.parseFloat(out.trim());
      if (Number.isFinite(seconds) && seconds > 0) {
        result.durationSeconds = seconds;
      }
    } catch {
      /* keep null */
    }
  }

  const ffmpeg = await firstWorking(binCandidates("ffmpeg", settings), [
    "-version",
  ]);
  if (ffmpeg) {
    const tmp = path.join(
      os.tmpdir(),
      `mask-studio-thumb-${process.pid}-${Date.now()}.jpg`,
    );
    const seek =
      result.durationSeconds !== null
        ? Math.min(3, result.durationSeconds * 0.1)
        : 0;
    try {
      await run(ffmpeg, [
        "-y",
        "-ss",
        seek.toFixed(2),
        "-i",
        videoPath,
        "-frames:v",
        "1",
        "-vf",
        "scale=192:-2",
        "-q:v",
        "7",
        tmp,
      ]);
      const jpeg = fs.readFileSync(tmp);
      result.thumbnail = `data:image/jpeg;base64,${jpeg.toString("base64")}`;
    } catch {
      /* keep null */
    } finally {
      fs.rmSync(tmp, { force: true });
    }
  }

  return result;
}
