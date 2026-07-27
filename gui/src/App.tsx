import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AppSettings,
  InferenceMode,
  JobSnapshot,
  PipelineDraft,
} from "../shared/types";
import { ConsolePanel } from "./components/ConsolePanel";
import { InspectorPanel } from "./components/InspectorPanel";
import type { InspectorActions } from "./components/InspectorPanel";
import { MonitorPanel } from "./components/MonitorPanel";
import { SourcePanel } from "./components/SourcePanel";
import type { SourceActions } from "./components/SourcePanel";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { useSplit } from "./hooks/useSplit";
import { desktopApi, isElectron } from "./lib/api";
import { browserSettings, emptyJob, loadDraft } from "./lib/defaults";
import { defaultBackend } from "./lib/models";

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

const SECTIONS_KEY = "mask-studio-sections";

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
  const [sections, setSections] = useState<Record<string, boolean>>(loadSections);
  const [toast, setToast] = useState<{ text: string; error: boolean } | null>(
    null,
  );
  const [tick, setTick] = useState(() => Date.now());
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const settingsLoaded = useRef(false);
  const sampled = useRef<{ id: string | null; frames: number }>({
    id: null,
    frames: -1,
  });

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

  const busy = ["validating", "running", "cancelling"].includes(job.status);
  const canRun =
    Boolean(draft.inputVideo.trim()) && Boolean(draft.outputRoot.trim()) && !busy;

  /* ── bootstrap + live job updates ─────────────────────────────────── */

  useEffect(() => {
    void desktopApi.bootstrap().then((data) => {
      setSettings(data.settings);
      setJob(data.job);
      settingsLoaded.current = true;
    });
    return desktopApi.onJobUpdate(setJob);
  }, []);

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

  useEffect(() => {
    window.localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
  }, [draft]);

  useEffect(() => {
    window.localStorage.setItem(SECTIONS_KEY, JSON.stringify(sections));
  }, [sections]);

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

  /* ── actions ──────────────────────────────────────────────────────── */

  const run = useCallback(
    async (dryRun: boolean) => {
      try {
        await desktopApi.saveSettings(settings);
        setJob(
          dryRun
            ? await desktopApi.validateWorkflow(draft, settings)
            : await desktopApi.startWorkflow(draft, settings),
        );
      } catch (error) {
        setToast({
          text: error instanceof Error ? error.message : "実行できませんでした。",
          error: true,
        });
      }
    },
    [draft, settings],
  );

  const cancel = useCallback(() => {
    void desktopApi.cancelWorkflow();
  }, []);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && busy) {
        cancel();
        return;
      }
      if (!(event.ctrlKey || event.metaKey)) {
        return;
      }
      if (event.key === "Enter" && canRun) {
        event.preventDefault();
        void run(false);
      } else if (event.key.toLowerCase() === "d" && canRun) {
        event.preventDefault();
        void run(true);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, canRun, cancel, run]);

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

  const sourceActions: SourceActions = useMemo(
    () => ({
      setInputVideo: (inputVideo) =>
        setDraft((current) => ({ ...current, inputVideo })),
      setOutputRoot: (outputRoot) =>
        setDraft((current) => ({ ...current, outputRoot })),
      setInferenceSqlite: (inputSqlite) => patchInference({ inputSqlite }),
      setTrackedSqlite: (trackedSqlite) => patchPostprocess({ trackedSqlite }),
      setFinalSqlite: (finalSqlite) => patchPostprocess({ finalSqlite }),
      pickVideo: () =>
        void pickInto("video", (inputVideo) =>
          setDraft((current) => ({ ...current, inputVideo })),
        ),
      pickOutput: () =>
        void desktopApi.pickDirectory().then((outputRoot) => {
          if (outputRoot) {
            setDraft((current) => ({ ...current, outputRoot }));
          }
        }),
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
      openOutput: () =>
        void desktopApi
          .openOutput(job.outputRoot ?? draft.outputRoot)
          .then((message) => {
            if (message) {
              setToast({ text: message, error: true });
            }
          }),
    }),
    [draft.outputRoot, job.outputRoot, patchInference, patchPostprocess, pickInto],
  );

  const inspectorActions: InspectorActions = useMemo(
    () => ({
      inference: patchInference,
      postprocess: patchPostprocess,
      overlay: patchOverlay,
      settings: (values: Partial<AppSettings>) =>
        setSettings((current) => ({ ...current, ...values })),
      changeMode: (mode: InferenceMode) => {
        patchInference({ mode });
        if (mode === "face") {
          patchPostprocess({ enabled: false });
          patchOverlay({
            raw: false,
            tracked: false,
            final: false,
            faces: true,
            finalIncludeFaces: false,
          });
        } else if (mode === "segmentation") {
          patchOverlay({ faces: false, finalIncludeFaces: false });
        } else {
          patchOverlay({ faces: true, finalIncludeFaces: true });
        }
      },
      changeModel: (segmentationModel) =>
        patchInference({
          segmentationModel,
          segmentationBackend: defaultBackend(segmentationModel),
        }),
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
    [patchInference, patchOverlay, patchPostprocess, pickInto],
  );

  /* ── derived ──────────────────────────────────────────────────────── */

  const elapsedSeconds = job.startedAt
    ? Math.max(
        job.telemetry.elapsedSeconds,
        ((job.completedAt ? new Date(job.completedAt).getTime() : tick) -
          new Date(job.startedAt).getTime()) /
          1_000,
      )
    : 0;

  const summary = useMemo(() => {
    switch (job.status) {
      case "running":
        return job.stage ? `${job.stage} を処理中` : "起動しています";
      case "cancelling":
        return "停止を要求しました";
      case "failed":
        return job.error ?? "処理に失敗しました";
      case "completed":
        return `${Object.keys(job.artifacts).length} 件の成果物`;
      case "validated":
        return "設定と入力は妥当です";
      case "cancelled":
        return "ジョブを中断しました";
      case "validating":
        return "設定を検証しています";
      default:
        return canRun ? "実行できます" : "動画と出力先を選択してください";
    }
  }, [canRun, job]);

  const shellStyle = {
    "--w-left": `${left.size}px`,
    "--w-right": `${right.size}px`,
  } as React.CSSProperties;

  return (
    <div
      className="app"
      style={{
        ...shellStyle,
        gridTemplateRows: `34px minmax(0, 1fr) 5px ${bottom.size}px 22px`,
      }}
    >
      <TopBar
        draft={draft}
        job={job}
        settings={settings}
        busy={busy}
        canRun={canRun}
        onRun={(dryRun) => void run(dryRun)}
        onCancel={cancel}
        onRuntime={() =>
          setSections((current) => ({ ...current, runtime: !current.runtime }))
        }
      />

      <div className="main">
        <SourcePanel draft={draft} job={job} busy={busy} actions={sourceActions} />
        <div {...left.handleProps} />
        <MonitorPanel
          draft={draft}
          job={job}
          elapsedSeconds={elapsedSeconds}
          statusLabel={STATUS_LABELS[job.status]}
          summary={summary}
          fpsHistory={fpsHistory}
        />
        <div {...right.handleProps} />
        <InspectorPanel
          draft={draft}
          settings={settings}
          busy={busy}
          open={sections}
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
