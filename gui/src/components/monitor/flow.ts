import type { JobSnapshot, PhaseProgress, PipelineDraft } from "../../../shared/types";
import { faceModelSpec, modelSpec } from "../../lib/models";
import type { NodeState } from "../../lib/stages";
import {
  CpuIcon,
  DatabaseIcon,
  EyeIcon,
  FilmIcon,
  LayersIcon,
  VideoIcon,
} from "../Icons";

export interface MonitorQueueInfo {
  total: number;
  pending: number;
  position: number;
  activeTitle: string | null;
}

export interface FlowItem {
  key: string;
  label: string;
  icon: typeof VideoIcon;
  state: NodeState;
  value: string;
  progress?: number | null;
  badge?: string;
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

/** Build the display-only pipeline graph from immutable workflow state. */
export function buildMonitorFlow(
  draft: PipelineDraft,
  job: JobSnapshot,
  queueInfo: MonitorQueueInfo,
): { nodes: FlowItem[]; overlayCount: number } {
  const overlayCount =
    draft.overlay.presets.length +
    [
      draft.overlay.raw,
      draft.overlay.tracked,
      draft.overlay.final,
      draft.overlay.faces,
    ].filter(Boolean).length;
  const nodes: FlowItem[] = [];
  const hasStarted = job.status !== "idle";
  const running = job.status === "running" || job.status === "cancelling";

  nodes.push({
    key: "input",
    label: "入力",
    icon: VideoIcon,
    state: hasStarted ? "done" : queueInfo.total > 0 ? "ready" : "waiting",
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
        : "ポリゴン";
    const cut = draft.postprocess.cutDetect
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
        : running && nodes.slice(1).every((node) => node.state === "done")
          ? "active"
          : "waiting";
  nodes.push({
    key: "output",
    label: "出力",
    icon: DatabaseIcon,
    state: outputState,
    value: draft.overlay.enabled && overlayCount > 0 ? "SQLite + 動画" : "SQLite",
    progress: outputState === "done" ? 1 : null,
  });

  return { nodes, overlayCount };
}
