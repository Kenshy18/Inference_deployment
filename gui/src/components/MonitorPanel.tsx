import { useEffect, useState } from "react";
import type {
  JobSnapshot,
  LivePreviewFrame,
  PhaseProgress,
  PipelineDraft,
} from "../../shared/types";
import { count, duration, rate } from "../lib/format";
import { desktopApi } from "../lib/api";
import { faceModelSpec, modelSpec } from "../lib/models";
import type { PipelineProgressEstimate } from "../lib/progress-estimator";
import { plannedStages, stageStates } from "../lib/stages";
import type { NodeState } from "../lib/stages";
import {
  CpuIcon,
  DatabaseIcon,
  EyeIcon,
  FilmIcon,
  LayersIcon,
  VideoIcon,
} from "./Icons";
import { Scope } from "./Scope";
import { Panel } from "./ui";

function previewPhaseLabel(phase: string): string {
  const labels: Record<string, string> = {
    segmentation_inference: "性器推論",
    face_inference: "顔推論",
    postprocess: "後処理",
  };
  return labels[phase] ?? phase;
}

function Metric({
  label,
  value,
  unit,
}: {
  label: string;
  value: string | null;
  unit?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <b
        className={value === null ? "is-null" : ""}
        title={value === null ? undefined : `${value}${unit ? ` ${unit}` : ""}`}
      >
        {value ?? "—"}
        {value !== null && unit && <em>{unit}</em>}
      </b>
    </div>
  );
}

function FlowNode({
  state,
  label,
  value,
  icon: Icon,
  step,
  progress,
  badge,
}: {
  state: NodeState;
  label: string;
  value: string;
  icon: typeof VideoIcon;
  step: number;
  progress?: number | null;
  badge?: string;
}) {
  const boundedProgress =
    progress === null || progress === undefined
      ? null
      : Math.min(100, Math.max(0, progress * 100));
  return (
    <div className={`flow__node is-${state}`} title={`${label} · ${value}`}>
      <div className="flow__node-head">
        <span className="flow__step">{String(step).padStart(2, "0")}</span>
        {badge && <em>{badge}</em>}
        <i aria-label={state} />
      </div>
      <div className="flow__node-body">
        <span className="flow__icon">
          <Icon />
        </span>
        <div>
          <b>{label}</b>
          <span>{value}</span>
        </div>
      </div>
      <div
        className={`flow__mini-progress ${
          state === "active" && boundedProgress === null
            ? "is-indeterminate"
            : ""
        }`}
      >
        <span
          style={
            boundedProgress === null
              ? undefined
              : { width: `${boundedProgress}%` }
          }
        />
      </div>
    </div>
  );
}

function phaseNodeState(phase: PhaseProgress): NodeState {
  if (phase.state === "failed") {
    return "failed";
  }
  if (phase.state === "running") {
    return "active";
  }
  if (phase.state === "complete") {
    return "done";
  }
  return "waiting";
}

function overlayExecutionLabel(
  mode: PipelineDraft["overlay"]["executionMode"],
): string {
  if (mode === "fast") {
    return "高速";
  }
  return mode === "nvenc" ? "NVENC" : "CPU";
}

function inferenceModeLabel(
  mode: PipelineDraft["inference"]["mode"],
): string {
  if (mode === "segmentation-face") {
    return "性器 + 顔";
  }
  return mode === "segmentation" ? "性器のみ" : "顔のみ";
}

function phaseDetail(value: string): string {
  if (!value) {
    return "待機中";
  }
  const labels: Record<string, string> = {
    "model-loading": "モデル準備中",
    frames: "フレーム処理",
    rendering: "描画・エンコード",
    concatenating: "セグメント結合",
    "preparing-workers": "ワーカー準備中",
    complete: "完了",
  };
  if (labels[value]) {
    return labels[value];
  }
  return value
    .replace(":input-validation", " · 入力検証")
    .replace(":running", " · 処理中")
    .replace(":output-validation", " · 出力検証")
    .replace(":complete", " · 完了")
    .replaceAll("_", " ");
}

function PhaseProgressRow({
  label,
  phase,
  enabled,
  unit,
}: {
  label: string;
  phase: PhaseProgress;
  enabled: boolean;
  unit: string;
}) {
  const progress = enabled ? phase.progress : 1;
  const percent =
    progress === null ? null : Math.min(100, Math.max(0, progress * 100));
  const state = enabled ? phase.state : "complete";
  const detail = enabled ? phaseDetail(phase.detail) : "対象外";
  const counts =
    enabled && phase.total !== null
      ? `${count(phase.completed)} / ${count(phase.total)} ${unit}`
      : enabled && phase.state === "running"
        ? "総量を確認中"
        : "—";
  return (
    <div className={`phase-progress is-${state}`}>
      <div className="phase-progress__head">
        <b>{label}</b>
        <strong>
          {percent === null
            ? "—"
            : `${phase.estimated ? "約 " : ""}${percent.toFixed(1)}%`}
        </strong>
      </div>
      <div
        className={`phase-progress__track ${
          enabled && phase.state === "running" && percent === null
            ? "is-indeterminate"
            : ""
        }`}
      >
        <span style={percent === null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="phase-progress__meta">
        <span title={phase.detail}>{detail}</span>
        <i>
          {counts}
          {phase.fps !== null ? ` · ${phase.fps.toFixed(1)} fps` : ""}
        </i>
      </div>
    </div>
  );
}

export interface QueueInfo {
  total: number;
  pending: number;
  position: number;
  activeTitle: string | null;
}

export interface HardwareHistories {
  gpu: number[];
  cpu: number[];
  vram: number[];
  memory: number[];
  temperature: number[];
}

export function MonitorPanel({
  draft,
  job,
  queueInfo,
  elapsedSeconds,
  progressEstimate,
  statusLabel,
  summary,
  fpsHistory,
  hardwareHistories,
}: {
  draft: PipelineDraft;
  job: JobSnapshot;
  queueInfo: QueueInfo;
  elapsedSeconds: number;
  progressEstimate: PipelineProgressEstimate;
  statusLabel: string;
  summary: string;
  fpsHistory: number[];
  hardwareHistories: HardwareHistories;
}) {
  const [activeTab, setActiveTab] = useState<"status" | "live">("status");
  const [preview, setPreview] = useState<LivePreviewFrame | null>(null);
  useEffect(() => desktopApi.onPreviewUpdate(setPreview), []);
  useEffect(() => setPreview(null), [job.id]);
  useEffect(() => {
    void desktopApi.setPreviewEnabled(activeTab === "live");
    return () => {
      void desktopApi.setPreviewEnabled(false);
    };
  }, [activeTab]);

  const stages = plannedStages(draft);
  const states = stageStates(stages, job);
  const stageFraction = job.telemetry.progress;
  const running = job.status === "running" || job.status === "cancelling";
  const overall = progressEstimate.overall;

  const overlayCount =
    draft.overlay.presets.length +
    [
      draft.overlay.raw,
      draft.overlay.tracked,
      draft.overlay.final,
      draft.overlay.faces,
    ].filter(Boolean).length;

  type FlowItem = {
    key: string;
    label: string;
    icon: typeof VideoIcon;
    state: NodeState;
    value: string;
    progress?: number | null;
    badge?: string;
  };
  const nodes: FlowItem[] = [];
  const hasStarted = job.status !== "idle";
  nodes.push({
    key: "input",
    label: "入力",
    icon: VideoIcon,
    state: hasStarted
      ? "done"
      : queueInfo.total > 0
        ? "ready"
        : "waiting",
    value:
      queueInfo.activeTitle ??
      (queueInfo.total > 0 ? `${queueInfo.total}本を待機` : "動画を選択"),
    progress: hasStarted ? 1 : null,
  });

  if (draft.inference.enabled) {
    const parallelBadge = draft.inference.parallelModels ? "並列" : undefined;
    if (draft.inference.mode !== "face") {
      const phase = job.telemetry.phases.segmentation_inference;
      nodes.push({
        key: "segmentation",
        label: "性器推論",
        icon: CpuIcon,
        state: phaseNodeState(phase),
        value: modelSpec(draft.inference.segmentationModel).label,
        progress: phase.progress,
        badge: parallelBadge,
      });
    }
    if (draft.inference.mode !== "segmentation") {
      const phase = job.telemetry.phases.face_inference;
      nodes.push({
        key: "face",
        label: "顔推論",
        icon: EyeIcon,
        state: phaseNodeState(phase),
        value: faceModelSpec(draft.inference.faceModel).label,
        progress: phase.progress,
        badge: parallelBadge,
      });
    }
  } else {
    nodes.push({
      key: "reuse",
      label: "推論結果",
      icon: DatabaseIcon,
      state: hasStarted ? "done" : "ready",
      value: "既存SQLiteを再利用",
      progress: hasStarted ? 1 : null,
      badge: "再利用",
    });
  }

  if (draft.postprocess.enabled) {
    const phase = job.telemetry.phases.postprocess;
    const policy =
      draft.postprocess.classPostprocessPolicySource === "editor"
        ? `${draft.postprocess.classPostprocessRules.length}クラス個別`
        : draft.postprocess.shapeMode === "ellipse"
          ? "楕円"
          : "ポリゴン";
    const cut =
      draft.postprocess.cutDetect
        ? draft.postprocess.precomputeCutsDuringInference
          ? " · カット先行"
          : " · カット"
        : "";
    nodes.push({
      key: "postprocess",
      label: "後処理",
      icon: LayersIcon,
      state: phaseNodeState(phase),
      value: `${policy}${cut}`,
      progress: phase.progress,
    });
  }

  if (draft.overlay.enabled && overlayCount > 0) {
    const phase = job.telemetry.phases.overlay;
    nodes.push({
      key: "overlay",
      label: "オーバーレイ",
      icon: FilmIcon,
      state: phaseNodeState(phase),
      value: `${overlayCount}本 · ${overlayExecutionLabel(
        draft.overlay.executionMode,
      )}`,
      progress: phase.progress,
    });
  }

  const outputState: NodeState =
    job.status === "completed"
      ? "done"
      : job.status === "failed"
        ? "failed"
        : running &&
            nodes.slice(1).every((node) => node.state === "done")
          ? "active"
          : "waiting";
  nodes.push({
    key: "output",
    label: "出力",
    icon: DatabaseIcon,
    state: outputState,
    value:
      draft.overlay.enabled && overlayCount > 0
        ? "SQLite + 動画"
        : "SQLite",
    progress: outputState === "done" ? 1 : null,
  });

  const activeNode = nodes.find((node) => node.state === "active");
  const enabledFaceInference =
    draft.inference.enabled && draft.inference.mode !== "segmentation";
  const facePhase = job.telemetry.phases.face_inference;
  const faceCountVisible =
    enabledFaceInference &&
    (facePhase.state === "running" ||
      facePhase.state === "complete" ||
      job.telemetry.faces > 0);

  return (
    <Panel
      title="Monitor"
      meta={inferenceModeLabel(draft.inference.mode)}
      actions={
        <>
          <div className="monitor-tabs" role="tablist" aria-label="Monitor view">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "status"}
              className={activeTab === "status" ? "is-active" : ""}
              onClick={() => setActiveTab("status")}
            >
              STATUS
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "live"}
              className={activeTab === "live" ? "is-active" : ""}
              onClick={() => setActiveTab("live")}
            >
              LIVE
            </button>
          </div>
          <span className="panel__meta">
            {job.dryRun ? "DRY RUN" : job.id ? `JOB ${job.id.slice(0, 19)}` : ""}
          </span>
        </>
      }
    >
      {activeTab === "status" ? (
        <>
      <div className="viewer">
        <div className="viewer__hud">
          <span>PIPELINE · {nodes.length} STEPS</span>
          <strong className="viewer__batch">
            BATCH&nbsp; {queueInfo.position} / {queueInfo.total}
          </strong>
          <span className="is-right">
            {statusLabel.toUpperCase()} · {activeNode?.label ?? "待機"}
          </span>
        </div>

        <div className="viewer__stack">
          <div className="flow">
            {nodes.map((node, index) => (
              <div key={node.key} style={{ display: "contents" }}>
                {index > 0 && (
                  <span
                    className={`flow__link ${
                      node.state === "done"
                        ? "is-done"
                        : node.state === "active"
                          ? "is-active"
                          : ""
                    }`}
                  />
                )}
                <FlowNode
                  state={node.state}
                  label={node.label}
                  value={node.value}
                  icon={node.icon}
                  step={index + 1}
                  progress={node.progress}
                  badge={node.badge}
                />
              </div>
            ))}
          </div>

          <div className="scope-grid">
            <Scope
              samples={fpsHistory}
              label="FPS"
              unit="fps"
              color="#5e8bff"
            />
            <Scope
              samples={hardwareHistories.gpu}
              label="GPU"
              unit="%"
              color="#a879ff"
              fixedMax={100}
            />
            <Scope
              samples={hardwareHistories.cpu}
              label="CPU"
              unit="%"
              color="#43c6ac"
              fixedMax={100}
            />
            <Scope
              samples={hardwareHistories.vram}
              label="VRAM"
              unit="%"
              color="#ff9f5a"
              fixedMax={100}
            />
            <Scope
              samples={hardwareHistories.memory}
              label="MEMORY"
              unit="%"
              color="#5ac8fa"
              fixedMax={100}
            />
            <Scope
              samples={hardwareHistories.temperature}
              label="GPU TEMP"
              unit="°C"
              color="#ff6577"
              fixedMax={100}
            />
          </div>

          <div className="viewer__footer">
            <div className="readout">
              <b className={overall === null ? "is-idle" : ""}>
                {overall === null ? "—" : `${(overall * 100).toFixed(1)}%`}
              </b>
              <div>
                <i>{summary}</i>
              </div>
            </div>
            <div className="eta eta--expanded">
              <div>
                <b>{duration(elapsedSeconds)}</b>
                <span>経過時間</span>
              </div>
              <div>
                <b>
                  {duration(progressEstimate.estimatedTotalSeconds)}
                </b>
                <span>予測所要</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="timeline">
        {stages.length === 0 ? (
          <div className="timeline__cell">
            <b>実行するステージがありません</b>
          </div>
        ) : (
          stages.map((stage, index) => {
            const state = states[index];
            const indeterminate = state === "active" && stageFraction === null;
            return (
              <div
                key={stage.id}
                className={`timeline__cell is-${state} ${
                  indeterminate ? "is-indeterminate" : ""
                }`}
                title={stage.id}
              >
                <span
                  className="timeline__fill"
                  style={
                    state === "active" && !indeterminate
                      ? { width: `${(stageFraction ?? 0) * 100}%` }
                      : undefined
                  }
                />
                <b>{stage.label}</b>
              </div>
            );
          })
        )}
      </div>

      <div className="phase-progress-grid">
        <PhaseProgressRow
          label="性器推論"
          phase={job.telemetry.phases.segmentation_inference}
          enabled={
            draft.inference.enabled &&
            draft.inference.mode !== "face"
          }
          unit="フレーム"
        />
        <PhaseProgressRow
          label="顔推論"
          phase={job.telemetry.phases.face_inference}
          enabled={
            draft.inference.enabled &&
            draft.inference.mode !== "segmentation"
          }
          unit="フレーム"
        />
        <PhaseProgressRow
          label="後処理"
          phase={job.telemetry.phases.postprocess}
          enabled={draft.postprocess.enabled}
          unit="ステージ"
        />
        <PhaseProgressRow
          label="オーバーレイ"
          phase={job.telemetry.phases.overlay}
          enabled={draft.overlay.enabled && overlayCount > 0}
          unit="フレーム"
        />
      </div>
        </>
      ) : (
        <div className="live-preview">
          <div className="live-preview__stage">
            {preview !== null && preview.jobId === job.id ? (
              <img
                src={preview.dataUrl}
                width={preview.width}
                height={preview.height}
                alt={`処理フレーム ${preview.frameIndex}`}
                draggable={false}
              />
            ) : (
              <div className="live-preview__empty">
                <span>LIVE PREVIEW</span>
                <b>処理結果を待っています</b>
                <i>最大5fps / 960 × 540 / 非同期プレビュー</i>
              </div>
            )}
            <div className="live-preview__scan" />
            <div className="live-preview__badge">
              <i /> LIVE
            </div>
          </div>
          <div className="live-preview__meta">
            <div>
              <span>PHASE</span>
              <b>
                {preview?.jobId === job.id
                  ? previewPhaseLabel(preview.phase)
                  : "待機中"}
              </b>
            </div>
            <div>
              <span>FRAME</span>
              <b>{preview?.jobId === job.id ? count(preview.frameIndex) : "—"}</b>
            </div>
            <div>
              <span>TIMECODE</span>
              <b>
                {preview?.jobId === job.id
                  ? duration(preview.timestampSeconds)
                  : "—"}
              </b>
            </div>
            <div>
              <span>{preview?.phase === "postprocess" ? "STAGE" : "MODEL"}</span>
              <b>
                {preview?.jobId === job.id
                  ? preview.model
                  : "—"}
              </b>
            </div>
            <div>
              <span>DETAIL</span>
              <b>
                {preview?.jobId === job.id
                  ? preview.detail || preview.status || "running"
                  : "—"}
              </b>
            </div>
            <div>
              <span>COALESCED</span>
              <b>{preview?.jobId === job.id ? count(preview.dropped) : "—"}</b>
            </div>
          </div>
        </div>
      )}

      <div className="metrics">
        <Metric label="wall fps" value={rate(job.telemetry.fps, 2)} />
        <Metric
          label="frames"
          value={
            job.telemetry.processedFrames === 0 && job.telemetry.totalFrames === null
              ? null
              : `${count(job.telemetry.processedFrames)}${
                  job.telemetry.totalFrames
                    ? ` / ${count(job.telemetry.totalFrames)}`
                    : ""
                }`
          }
        />
        <Metric
          label="detections"
          value={job.telemetry.detections ? count(job.telemetry.detections) : null}
        />
        <Metric
          label="masks"
          value={job.telemetry.masks ? count(job.telemetry.masks) : null}
        />
        <Metric
          label="faces"
          value={faceCountVisible ? count(job.telemetry.faces) : null}
        />
        <Metric label="device" value={draft.inference.device || null} />
        <Metric
          label="codec"
          value={draft.overlay.enabled ? draft.overlay.codec : null}
        />
      </div>
    </Panel>
  );
}
