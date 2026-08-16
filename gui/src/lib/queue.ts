import type { PipelineDraft, QueueItem } from "../../shared/types";
import { faceModelSpec, modelSpec } from "./models";

export const QUEUE_STORAGE_KEY = "mask-studio-queue";

const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "mkv", "avi", "webm"]);

export function isVideoPath(value: string): boolean {
  const ext = value.split(".").at(-1)?.toLowerCase() ?? "";
  return VIDEO_EXTENSIONS.has(ext);
}

/** 1-based current position while running, or the number already attempted
 *  while idle. Failed items count because the sequential runner has moved on. */
export function batchPosition(
  queue: ReadonlyArray<Pick<QueueItem, "status">>,
): number {
  const settled = queue.filter(
    (item) => item.status === "done" || item.status === "failed",
  ).length;
  const running = queue.some((item) => item.status === "processing") ? 1 : 0;
  return Math.min(queue.length, settled + running);
}

/** File name without directories and extension — the queue display title. */
export function titleFromPath(value: string): string {
  const base = value.split(/[\\/]/).filter(Boolean).at(-1) ?? value;
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function pathSeparator(root: string): string {
  return root.includes("\\") && !root.includes("/") ? "\\" : "/";
}

export function joinPath(root: string, name: string): string {
  const sep = pathSeparator(root);
  const trimmed = root.replace(/[\\/]+$/, "");
  return `${trimmed}${sep}${name}`;
}

function sanitizeName(value: string): string {
  const cleaned = value.replaceAll(/[\\/:*?"<>|]/g, "_").trim();
  return cleaned || "video";
}

/** Job folder inside the output repository: the video title, made unique
 *  against every folder the queue already claimed. */
export function uniqueOutputDir(
  repoRoot: string,
  title: string,
  takenDirs: Iterable<string | null>,
): string {
  const taken = new Set(
    [...takenDirs]
      .filter((value): value is string => Boolean(value))
      .map((value) => value.toLowerCase()),
  );
  const stem = sanitizeName(title);
  for (let index = 1; ; index += 1) {
    const name = index === 1 ? stem : `${stem}_${index}`;
    const dir = joinPath(repoRoot, name);
    if (!taken.has(dir.toLowerCase())) {
      return dir;
    }
  }
}

/** One-line inference summary shown on processed queue entries. */
export function settingsSummary(draft: PipelineDraft): string {
  const { inference, postprocess, overlay } = draft;
  const parts: string[] = [];
  if (!inference.enabled) {
    parts.push("既存SQLite");
  } else {
    const models: string[] = [];
    if (inference.mode !== "face") {
      models.push(modelSpec(inference.segmentationModel).label);
    }
    if (inference.mode !== "segmentation") {
      models.push(faceModelSpec(inference.faceModel).label);
    }
    parts.push(models.join(" + "));
  }
  if (inference.mode !== "face" && postprocess.enabled) {
    parts.push("ポリゴン");
  }
  parts.push(
    overlay.enabled ? `overlay ${overlay.executionMode}` : "overlayなし",
  );
  return parts.join(" · ");
}

export function loadQueue(): QueueItem[] {
  try {
    const saved = JSON.parse(
      window.localStorage.getItem(QUEUE_STORAGE_KEY) ?? "null",
    ) as QueueItem[] | null;
    if (!Array.isArray(saved)) {
      return [];
    }
    return saved
      .filter((item) => item && item.id && item.path)
      .map((item) => {
        const legacyOutput =
          item.status === "done" && item.outputDir
            ? [
                {
                  id: `legacy:${item.id}:${item.outputDir}`,
                  outputDir: item.outputDir,
                  summary: item.summary ?? null,
                  completedAt: item.completedAt ?? null,
                  artifactCount: item.artifactCount ?? null,
                },
              ]
            : [];
        return {
          ...item,
          width: item.width ?? null,
          height: item.height ?? null,
          fps: item.fps ?? null,
          frameCount: item.frameCount ?? null,
          completedAt: item.completedAt ?? null,
          artifactCount: item.artifactCount ?? null,
          outputs:
            Array.isArray(item.outputs) && item.outputs.length > 0
              ? item.outputs
              : legacyOutput,
        };
      });
  } catch {
    return [];
  }
}

export function saveQueue(queue: QueueItem[]): void {
  window.localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(queue));
}

export function newQueueItem(path: string): QueueItem {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    path,
    title: titleFromPath(path),
    durationSeconds: null,
    width: null,
    height: null,
    fps: null,
    frameCount: null,
    thumbnail: null,
    status: "pending",
    outputDir: null,
    summary: null,
    completedAt: null,
    artifactCount: null,
    outputs: [],
    error: null,
  };
}
