import type { PipelineDraft, QueueItem } from "../../shared/types";
import { faceModelSpec, modelSpec } from "./models";

export const QUEUE_STORAGE_KEY = "mask-studio-queue";

const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "mkv", "avi", "webm"]);

export function isVideoPath(value: string): boolean {
  const ext = value.split(".").at(-1)?.toLowerCase() ?? "";
  return VIDEO_EXTENSIONS.has(ext);
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
    const name = index === 1 ? stem : `${stem}-${index}`;
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
    parts.push(postprocess.shapeMode === "ellipse" ? "楕円" : "ポリゴン");
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
    return saved.filter((item) => item && item.id && item.path);
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
    thumbnail: null,
    status: "pending",
    outputDir: null,
    summary: null,
    error: null,
  };
}
