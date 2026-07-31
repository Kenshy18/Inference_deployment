import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AppSettings,
  InferenceMode,
  JobSnapshot,
  OverlayExecutionMode,
  PipelineDraft,
  QueueItem,
  SettingsView,
} from "../shared/types";
import { ConsolePanel } from "./components/ConsolePanel";
import { InspectorPanel } from "./components/InspectorPanel";
import type { InspectorActions } from "./components/InspectorPanel";
import {
  MonitorPanel,
  type HardwareHistories,
} from "./components/MonitorPanel";
import { SourcePanel } from "./components/SourcePanel";
import type { QueueActions } from "./components/SourcePanel";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { useSplit } from "./hooks/useSplit";
import { desktopApi, isElectron } from "./lib/api";
import {
  browserSettings,
  DRAFT_STORAGE_VERSION,
  emptyJob,
  loadDraft,
} from "./lib/defaults";
import { defaultBackend, defaultFaceBackend } from "./lib/models";
import {
  batchPosition,
  isVideoPath,
  loadQueue,
  newQueueItem,
  saveQueue,
  settingsSummary,
  uniqueOutputDir,
} from "./lib/queue";
import { estimatePipelineProgress } from "./lib/progress-estimator";

const STATUS_LABELS: Record<JobSnapshot["status"], string> = {
  idle: "待機中",
  validating: "検証中",
  validated: "検証済み",
  running: "実行中",
  cancelling: "停止中",
  cancelled: "キャンセル",
  completed: "完了",
  failed: "失敗",
};

const BUSY_STATUSES: ReadonlyArray<JobSnapshot["status"]> = [
  "validating",
  "running",
  "cancelling",
];

const SECTIONS_KEY = "mask-studio-sections";
const SETTINGS_VIEW_KEY = "mask-studio-settings-view";
const TELEMETRY_POINTS = 180;

const EMPTY_HARDWARE_HISTORIES: HardwareHistories = {
  gpu: [],
  cpu: [],
  vram: [],
  memory: [],
  temperature: [],
};

function appendSample(history: number[], value: number | null): number[] {
  return value === null
    ? history
    : [...history, value].slice(-TELEMETRY_POINTS);
}

function loadSettingsView(): SettingsView {
  return window.localStorage.getItem(SETTINGS_VIEW_KEY) === "advanced"
    ? "advanced"
    : "simple";
}

function loadSections(): Record<string, boolean> {
  const fallback = {
    inference: true,
    postprocess: true,
    overlay: true,
    runtime: false,
  };
  try {
    const saved = JSON.parse(
      window.localStorage.getItem(SECTIONS_KEY) ?? "null",
    ) as Record<string, boolean> | null;
    return saved ? { ...fallback, ...saved } : fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [draft, setDraft] = useState<PipelineDraft>(loadDraft);
  const [settings, setSettings] = useState<AppSettings>(browserSettings);
  const [job, setJob] = useState<JobSnapshot>(emptyJob);
  const [queue, setQueue] = useState<QueueItem[]>(loadQueue);
  const [sections, setSections] = useState<Record<string, boolean>>(loadSections);
  const [settingsView, setSettingsView] =
    useState<SettingsView>(loadSettingsView);
  const [toast, setToast] = useState<{ text: string; error: boolean } | null>(
    null,
  );
  const [tick, setTick] = useState(() => Date.now());
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const [hardwareHistories, setHardwareHistories] =
    useState<HardwareHistories>(EMPTY_HARDWARE_HISTORIES);
  const settingsLoaded = useRef(false);
  const sampled = useRef<{ id: string | null; frames: number }>({
    id: null,
    frames: -1,
  });

  /* Queue orchestration lives on refs so the job-update effect and the
     sequential runner always see current state without re-subscribing. */
  const draftRef = useRef(draft);
  const settingsRef = useRef(settings);
  const queueRef = useRef(queue);
  const autoRunRef = useRef(false);
  const removeAfterCancelRef = useRef<string | null>(null);
  const lastTransitionRef = useRef("");
  const probingRef = useRef(new Set<string>());

  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);
  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);
  useEffect(() => {
    queueRef.current = queue;
    saveQueue(queue);
  }, [queue]);

  const left = useSplit("mask-studio-split-left", 250, {
    min: 200,
    max: 420,
    side: "start",
    axis: "x",
  });
  const right = useSplit("mask-studio-split-right", 296, {
    min: 240,
    max: 460,
    side: "end",
    axis: "x",
  });
  const bottom = useSplit("mask-studio-split-console", 168, {
    min: 46,
    max: 560,
    side: "end",
    axis: "y",
  });

  const busy = BUSY_STATUSES.includes(job.status);
  const pendingCount = queue.filter(
    (item) => item.status === "pending",
  ).length;
  const canRun =
    pendingCount > 0 && Boolean(draft.outputRoot.trim()) && !busy;

  const patchItem = useCallback(
    (id: string, values: Partial<QueueItem>) =>
      setQueue((current) =>
        current.map((item) =>
          item.id === id ? { ...item, ...values } : item,
        ),
      ),
    [],
  );

  /* ── sequential queue runner ──────────────────────────────────────── */

  const startItem = useCallback(
    async (item: QueueItem) => {
      const currentDraft = draftRef.current;
      const outputDir = uniqueOutputDir(
        currentDraft.outputRoot,
        item.title,
        queueRef.current.flatMap((entry) => [
          entry.id === item.id ? null : entry.outputDir,
          ...entry.outputs.map((output) => output.outputDir),
        ]),
      );
      const summary = settingsSummary(currentDraft);
      patchItem(item.id, {
        status: "processing",
        outputDir,
        summary,
        error: null,
      });
      try {
        await desktopApi.saveSettings(settingsRef.current);
        const started = await desktopApi.startWorkflow(
          { ...currentDraft, inputVideo: item.path, outputRoot: outputDir },
          settingsRef.current,
        );
        if (started.outputRoot && started.outputRoot !== outputDir) {
          patchItem(item.id, { outputDir: started.outputRoot });
        }
        setJob(started);
      } catch (error) {
        const text =
          error instanceof Error ? error.message : "実行できませんでした。";
        patchItem(item.id, { status: "failed", error: text });
        autoRunRef.current = false;
        setToast({ text, error: true });
      }
    },
    [patchItem],
  );

  const startNext = useCallback(() => {
    const next = queueRef.current.find((item) => item.status === "pending");
    if (next && autoRunRef.current) {
      void startItem(next);
    } else {
      autoRunRef.current = false;
    }
  }, [startItem]);

  const runQueue = useCallback(() => {
    const next = queueRef.current.find((item) => item.status === "pending");
    if (!next || !draftRef.current.outputRoot.trim()) {
      return;
    }
    autoRunRef.current = true;
    void startItem(next);
  }, [startItem]);

  const dryRun = useCallback(async () => {
    const next = queueRef.current.find((item) => item.status === "pending");
    if (!next || !draftRef.current.outputRoot.trim()) {
      return;
    }
    const outputDir = uniqueOutputDir(
      draftRef.current.outputRoot,
      next.title,
      queueRef.current.flatMap((entry) => [
        entry.outputDir,
        ...entry.outputs.map((output) => output.outputDir),
      ]),
    );
    try {
      await desktopApi.saveSettings(settingsRef.current);
      setJob(
        await desktopApi.validateWorkflow(
          {
            ...draftRef.current,
            inputVideo: next.path,
            outputRoot: outputDir,
          },
          settingsRef.current,
        ),
      );
    } catch (error) {
      setToast({
        text: error instanceof Error ? error.message : "検証できませんでした。",
        error: true,
      });
    }
  }, []);

  const cancelAll = useCallback(() => {
    autoRunRef.current = false;
    removeAfterCancelRef.current = null;
    void desktopApi.cancelWorkflow();
  }, []);

  /* React to terminal job states: advance, fail, or unwind the queue. */
  useEffect(() => {
    if (!job.id || job.dryRun) {
      return;
    }
    const key = `${job.id}:${job.status}`;
    if (lastTransitionRef.current === key) {
      return;
    }
    if (!["completed", "failed", "cancelled"].includes(job.status)) {
      return;
    }
    lastTransitionRef.current = key;
    const active = queueRef.current.find(
      (item) => item.status === "processing",
    );
    if (!active) {
      return;
    }
    if (job.status === "completed") {
      const completedOutput = job.outputRoot ?? active.outputDir;
      patchItem(active.id, {
        status: "done",
        completedAt: job.completedAt,
        artifactCount: Object.keys(job.artifacts).length,
        outputs:
          completedOutput === null
            ? active.outputs
            : [
                ...active.outputs,
                {
                  id: job.id ?? `${Date.now()}`,
                  outputDir: completedOutput,
                  summary: active.summary,
                  completedAt: job.completedAt,
                  artifactCount: Object.keys(job.artifacts).length,
                },
              ],
        error: null,
      });
      startNext();
    } else if (job.status === "failed") {
      patchItem(active.id, {
        status: "failed",
        error: job.error ?? "処理に失敗しました",
      });
      startNext();
    } else if (removeAfterCancelRef.current === active.id) {
      removeAfterCancelRef.current = null;
      setQueue((current) => current.filter((item) => item.id !== active.id));
      startNext();
    } else {
      patchItem(active.id, {
        status: "pending",
        outputDir: null,
        summary: null,
      });
      autoRunRef.current = false;
    }
  }, [job, patchItem, startNext]);

  /* ── bootstrap + live job updates ─────────────────────────────────── */

  useEffect(() => {
    void desktopApi.bootstrap().then((data) => {
      setSettings(data.settings);
      setJob(data.job);
      settingsLoaded.current = true;
      if (!BUSY_STATUSES.includes(data.job.status)) {
        /* A reload lost the runner loop; re-queue anything left mid-flight. */
        setQueue((current) =>
          current.map((item) =>
            item.status === "processing"
              ? { ...item, status: "pending", outputDir: null, summary: null }
              : item,
          ),
        );
      }
    });
    return desktopApi.onJobUpdate(setJob);
  }, []);

  /* Refresh queue entries saved by older GUI versions that only stored a
     duration. Resolution/FPS/frame count feed the wall-clock predictor. */
  useEffect(() => {
    if (!settingsLoaded.current) {
      return;
    }
    for (const item of queue) {
      if (
        (item.width !== null &&
          item.height !== null &&
          item.fps !== null &&
          item.frameCount !== null) ||
        probingRef.current.has(item.id)
      ) {
        continue;
      }
      probingRef.current.add(item.id);
      void desktopApi
        .probeVideo(item.path, settings)
        .then((probe) =>
          patchItem(item.id, {
            durationSeconds: probe.durationSeconds,
            width: probe.width,
            height: probe.height,
            fps: probe.fps,
            frameCount: probe.frameCount,
            thumbnail: probe.thumbnail ?? item.thumbnail,
          }),
        )
        .catch(() => undefined);
    }
  }, [patchItem, queue, settings]);

  /* One throughput sample per frame-count change, so the scope traces the run
     rather than the log volume. */
  useEffect(() => {
    if (job.id !== sampled.current.id) {
      sampled.current = { id: job.id, frames: -1 };
      setFpsHistory([]);
    }
    const { fps, processedFrames } = job.telemetry;
    if (fps !== null && processedFrames !== sampled.current.frames) {
      sampled.current.frames = processedFrames;
      setFpsHistory((current) => [...current, fps].slice(-180));
    }
  }, [job]);

  /* Hardware telemetry is intentionally independent from workflow logs. A
     one-second, non-overlapping poll is responsive enough for a resource graph
     and has negligible impact on inference or log size. */
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      try {
        const sample = await desktopApi.sampleHardware();
        if (!cancelled) {
          setHardwareHistories((current) => ({
            gpu: appendSample(current.gpu, sample.gpuPercent),
            cpu: appendSample(current.cpu, sample.cpuPercent),
            vram: appendSample(current.vram, sample.vramPercent),
            memory: appendSample(current.memory, sample.memoryPercent),
            temperature: appendSample(
              current.temperature,
              sample.gpuTemperatureC,
            ),
          }));
        }
      } catch {
        // Monitoring must never interrupt a workflow.
      }
      if (!cancelled) {
        timer = window.setTimeout(() => void poll(), 1_000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
    window.localStorage.setItem(
      "mask-studio-draft-version",
      DRAFT_STORAGE_VERSION,
    );
  }, [draft]);

  useEffect(() => {
    window.localStorage.setItem(SECTIONS_KEY, JSON.stringify(sections));
  }, [sections]);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_VIEW_KEY, settingsView);
  }, [settingsView]);

  /* Runtime settings persist on their own, so a run never uses stale paths. */
  useEffect(() => {
    if (!settingsLoaded.current) {
      return;
    }
    const timer = window.setTimeout(() => {
      void desktopApi.saveSettings(settings).catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [settings]);

  useEffect(() => {
    if (!toast) {
      return;
    }
    const timer = window.setTimeout(() => setToast(null), 4_000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!busy) {
      setTick(Date.now());
      return;
    }
    const timer = window.setInterval(() => setTick(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && busy) {
        cancelAll();
        return;
      }
      if (!(event.ctrlKey || event.metaKey)) {
        return;
      }
      if (event.key === "Enter" && canRun) {
        event.preventDefault();
        runQueue();
      } else if (event.key.toLowerCase() === "d" && canRun) {
        event.preventDefault();
        void dryRun();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, canRun, cancelAll, dryRun, runQueue]);

  /* ── queue actions ────────────────────────────────────────────────── */

  const addPaths = useCallback(
    (paths: string[]) => {
      const videos = paths.filter(Boolean).filter(isVideoPath);
      if (videos.length === 0) {
        if (paths.length > 0) {
          setToast({ text: "動画ファイルではありません。", error: true });
        }
        return;
      }
      const active = new Set(
        queueRef.current
          .filter(
            (item) =>
              item.status === "pending" || item.status === "processing",
          )
          .map((item) => item.path),
      );
      const fresh = videos.filter((path) => !active.has(path));
      if (fresh.length === 0) {
        setToast({ text: "既にキューへ追加済みです。", error: false });
        return;
      }
      const items = fresh.map(newQueueItem);
      setQueue((current) => [...current, ...items]);
      for (const item of items) {
        probingRef.current.add(item.id);
        void desktopApi
          .probeVideo(item.path, settingsRef.current)
          .then((probe) =>
            patchItem(item.id, {
              durationSeconds: probe.durationSeconds,
              width: probe.width,
              height: probe.height,
              fps: probe.fps,
              frameCount: probe.frameCount,
              thumbnail: probe.thumbnail,
            }),
          )
          .catch(() => undefined);
      }
    },
    [patchItem],
  );

  const queueActions: QueueActions = useMemo(
    () => ({
      addFiles: (files) =>
        addPaths(
          files.map(
            (file) => desktopApi.pathForFile(file) ?? file.name,
          ),
        ),
      addByPicker: () =>
        void desktopApi.pickVideos().then((paths) => addPaths(paths)),
      remove: (id) =>
        setQueue((current) =>
          current.filter(
            (item) => item.id !== id || item.status === "processing",
          ),
        ),
      stopAndRemove: (id) => {
        const item = queueRef.current.find((entry) => entry.id === id);
        if (!item || item.status !== "processing") {
          return;
        }
        removeAfterCancelRef.current = id;
        void desktopApi.cancelWorkflow();
      },
      requeue: (id) =>
        patchItem(id, {
          status: "pending",
          outputDir: null,
          summary: null,
          completedAt: null,
          artifactCount: null,
          error: null,
        }),
      openOutput: (id) => {
        const item = queueRef.current.find((entry) => entry.id === id);
        if (!item?.outputDir) {
          return;
        }
        void desktopApi.openOutput(item.outputDir).then((message) => {
          if (message) {
            setToast({ text: message, error: true });
          }
        });
      },
      openOutputPath: (outputPath) => {
        void desktopApi.openOutput(outputPath).then((message) => {
          if (message) {
            setToast({ text: message, error: true });
          }
        });
      },
      setOutputRoot: (outputRoot) =>
        setDraft((current) => ({ ...current, outputRoot })),
      pickOutput: () =>
        void desktopApi.pickDirectory().then((outputRoot) => {
          if (outputRoot) {
            setDraft((current) => ({ ...current, outputRoot }));
          }
        }),
    }),
    [addPaths, patchItem],
  );

  /* ── draft actions ────────────────────────────────────────────────── */

  const patchInference = useCallback(
    (values: Partial<PipelineDraft["inference"]>) =>
      setDraft((current) => ({
        ...current,
        inference: { ...current.inference, ...values },
      })),
    [],
  );

  const patchPostprocess = useCallback(
    (values: Partial<PipelineDraft["postprocess"]>) =>
      setDraft((current) => ({
        ...current,
        postprocess: { ...current.postprocess, ...values },
      })),
    [],
  );

  const patchOverlay = useCallback(
    (values: Partial<PipelineDraft["overlay"]>) =>
      setDraft((current) => ({
        ...current,
        overlay: { ...current.overlay, ...values },
      })),
    [],
  );

  const patchExecution = useCallback(
    (values: Partial<PipelineDraft["execution"]>) =>
      setDraft((current) => ({
        ...current,
        execution: { ...current.execution, ...values },
      })),
    [],
  );

  const pickInto = useCallback(
    async (
      kind: "video" | "sqlite" | "python",
      apply: (value: string) => void,
    ) => {
      const selected = await desktopApi.pickFile(kind);
      if (selected) {
        apply(selected);
      }
    },
    [],
  );

  const inspectorActions: InspectorActions = useMemo(
    () => ({
      inference: patchInference,
      postprocess: patchPostprocess,
      overlay: patchOverlay,
      execution: patchExecution,
      settings: (values: Partial<AppSettings>) =>
        setSettings((current) => ({ ...current, ...values })),
      pickSqlite: (target) =>
        void pickInto("sqlite", (value) => {
          if (target === "inference") {
            patchInference({ inputSqlite: value });
          } else if (target === "tracked") {
            patchPostprocess({ trackedSqlite: value });
          } else {
            patchPostprocess({ finalSqlite: value });
          }
        }),
      changeMode: (mode: InferenceMode) => {
        patchInference({
          mode,
          parallelModels:
            mode === "segmentation-face"
              ? draft.inference.parallelModels
              : false,
          parallelModelStaggerSeconds:
            mode === "segmentation-face"
              ? draft.inference.parallelModelStaggerSeconds
              : 0,
        });
        if (mode === "face") {
          patchPostprocess({
            enabled: false,
            faceMaskTarget:
              draft.inference.faceModel === "face_dino_v2" ? "eyes" : "none",
            precomputeCutsDuringInference: draft.postprocess.cutDetect,
          });
          patchOverlay({
            raw: false,
            tracked: false,
            final: false,
            faces: true,
            finalIncludeFaces: false,
            presets: ["face-simple"],
            faceMaskTarget: "none",
          });
        } else if (mode === "segmentation") {
          patchPostprocess({
            enabled: true,
            faceMaskTarget: "none",
            precomputeCutsDuringInference: draft.postprocess.cutDetect,
          });
          patchOverlay({
            faces: false,
            finalIncludeFaces: false,
            presets: ["genital-simple"],
            faceMaskTarget: "none",
          });
        } else {
          patchPostprocess({
            enabled: true,
            faceMaskTarget:
              draft.inference.faceModel === "face_dino_v2" ? "eyes" : "none",
            precomputeCutsDuringInference: draft.postprocess.cutDetect,
          });
          patchOverlay({
            faces: false,
            finalIncludeFaces: false,
            presets: ["combined-simple"],
            faceMaskTarget: "none",
          });
        }
      },
      changeModel: (segmentationModel) =>
        patchInference({
          segmentationModel,
          segmentationBackend: defaultBackend(segmentationModel),
          parallelModels:
            segmentationModel === "dinov3_codino_mh0"
              ? draft.inference.parallelModels
              : false,
          parallelModelStaggerSeconds:
            segmentationModel === "dinov3_codino_mh0"
              ? draft.inference.parallelModelStaggerSeconds
              : 0,
        }),
      changeFaceModel: (faceModel) => {
        patchInference({
          faceModel,
          faceBackend: defaultFaceBackend(faceModel),
          faceTrtBundle:
            faceModel === "face_dino_v2"
              ? draft.inference.faceTrtBundle
              : "",
          parallelModels:
            faceModel === "face_dino_v2"
              ? draft.inference.parallelModels
              : false,
          parallelModelStaggerSeconds:
            faceModel === "face_dino_v2"
              ? draft.inference.parallelModelStaggerSeconds
              : 0,
        });
        if (faceModel !== "face_dino_v2") {
          patchPostprocess({
            faceMaskTarget: "none",
            precomputeCutsDuringInference:
              draft.postprocess.cutDetect,
          });
          patchOverlay({ faceMaskTarget: "none" });
        } else if (draft.inference.mode !== "segmentation") {
          patchPostprocess({
            faceMaskTarget: "eyes",
            precomputeCutsDuringInference: draft.postprocess.cutDetect,
          });
          patchOverlay({ faceMaskTarget: "none" });
        }
      },
      changeOverlayExecution: (executionMode: OverlayExecutionMode) => {
        if (executionMode === "cpu") {
          patchOverlay({
            executionMode,
            codec: "h264",
            targetBitrateMbps: null,
            copyAudio: false,
            faststart: false,
            cpuWorkers: 0,
          });
        } else if (executionMode === "nvenc") {
          patchOverlay({
            executionMode,
            codec: "h264_nvenc",
            targetBitrateMbps: null,
            nvencPreset: "p5",
            copyAudio: false,
            faststart: false,
            cpuWorkers: 0,
          });
        } else {
          patchOverlay({
            executionMode,
            codec: "h264_nvenc",
            targetBitrateMbps: draft.overlay.targetBitrateMbps ?? 8,
            nvencPreset: "p1",
          });
        }
      },
      pickBackendRoot: () =>
        void desktopApi.pickDirectory().then((backendRoot) => {
          if (backendRoot) {
            setSettings((current) => ({ ...current, backendRoot }));
          }
        }),
      pickPython: () =>
        void pickInto("python", (runtimePython) =>
          setSettings((current) => ({ ...current, runtimePython })),
        ),
    }),
    [
      draft.inference.parallelModelStaggerSeconds,
      draft.inference.parallelModels,
      draft.inference.faceModel,
      draft.inference.faceTrtBundle,
      draft.inference.mode,
      draft.overlay.targetBitrateMbps,
      draft.postprocess.cutDetect,
      patchExecution,
      patchInference,
      patchOverlay,
      patchPostprocess,
      pickInto,
    ],
  );

  /* ── derived ──────────────────────────────────────────────────────── */

  const activeItem = queue.find((item) => item.status === "processing") ?? null;
  const currentBatchPosition = batchPosition(queue);
  const monitoredItem =
    activeItem ?? queue.find((item) => item.status === "pending") ?? null;

  const elapsedSeconds = job.startedAt
    ? Math.max(
        job.telemetry.elapsedSeconds,
        ((job.completedAt ? new Date(job.completedAt).getTime() : tick) -
          new Date(job.startedAt).getTime()) /
          1_000,
      )
    : 0;
  const progressEstimate = useMemo(
    () =>
      estimatePipelineProgress(
        draft,
        job,
        monitoredItem,
        elapsedSeconds,
      ),
    [draft, elapsedSeconds, job, monitoredItem],
  );

  const summary = useMemo(() => {
    switch (job.status) {
      case "running":
        return job.stage
          ? `${activeItem ? `${activeItem.title} — ` : ""}${job.stage} を処理中`
          : "起動しています";
      case "cancelling":
        return "停止を要求しました";
      case "failed":
        return job.error ?? "処理に失敗しました";
      case "completed":
        return pendingCount > 0
          ? `完了 — 残り${pendingCount}本`
          : `${Object.keys(job.artifacts).length} 件の成果物`;
      case "validated":
        return "設定と入力は妥当です";
      case "cancelled":
        return "ジョブを中断しました";
      case "validating":
        return "設定を検証しています";
      default:
        if (queue.length === 0) {
          return "キューに動画を追加してください";
        }
        if (!draft.outputRoot.trim()) {
          return "出力リポジトリを選択してください";
        }
        return pendingCount > 0
          ? `${pendingCount}本を実行できます`
          : "キューは全て処理済みです";
    }
  }, [activeItem, draft.outputRoot, job, pendingCount, queue.length]);

  const shellStyle = {
    "--w-left": `${left.size}px`,
    "--w-right": `${right.size}px`,
  } as React.CSSProperties;

  return (
    <div
      className="app"
      style={{
        ...shellStyle,
        gridTemplateRows: `46px minmax(0, 1fr) 6px ${bottom.size}px 26px`,
      }}
    >
      <TopBar
        queueTotal={queue.length}
        queuePending={pendingCount}
        outputRoot={draft.outputRoot}
        job={job}
        busy={busy}
        canRun={canRun}
        onRun={(isDryRun) => (isDryRun ? void dryRun() : runQueue())}
        onCancel={cancelAll}
      />

      <div className="main">
        <SourcePanel
          queue={queue}
          outputRoot={draft.outputRoot}
          busy={busy}
          activeProgress={progressEstimate.overall}
          actions={queueActions}
        />
        <div {...left.handleProps} />
        <MonitorPanel
          draft={draft}
          job={job}
          queueInfo={{
            total: queue.length,
            pending: pendingCount,
            position: currentBatchPosition,
            activeTitle: activeItem?.title ?? null,
          }}
          elapsedSeconds={elapsedSeconds}
          progressEstimate={progressEstimate}
          statusLabel={STATUS_LABELS[job.status]}
          summary={summary}
          fpsHistory={fpsHistory}
          hardwareHistories={hardwareHistories}
        />
        <div {...right.handleProps} />
        <InspectorPanel
          draft={draft}
          settings={settings}
          busy={busy}
          open={sections}
          viewMode={settingsView}
          onViewModeChange={setSettingsView}
          onToggle={(key) =>
            setSections((current) => ({ ...current, [key]: !current[key] }))
          }
          actions={inspectorActions}
        />
      </div>

      <div {...bottom.handleProps} />
      <ConsolePanel job={job} />

      <StatusBar
        job={job}
        settings={settings}
        statusLabel={STATUS_LABELS[job.status]}
        elapsedSeconds={elapsedSeconds}
        isElectron={isElectron}
      />

      {toast && (
        <div className={`toast ${toast.error ? "is-error" : ""}`}>{toast.text}</div>
      )}
    </div>
  );
}
