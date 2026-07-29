import type { JobSnapshot, PipelineDraft } from "../../shared/types";

/** A stage the orchestration runner will execute, in order.
 *  `id` matches the `[stage]` prefix the runner prints on every log line. */
export interface PlannedStage {
  id: string;
  label: string;
}

export function plannedStages(draft: PipelineDraft): PlannedStage[] {
  const stages: PlannedStage[] = [];
  if (draft.inference.enabled) {
    stages.push({ id: "inference", label: "inference" });
  }
  if (draft.postprocess.enabled) {
    stages.push({ id: "postprocess", label: "postprocess" });
  }
  if (draft.overlay.enabled) {
    for (const preset of draft.overlay.presets) {
      const name = preset.replaceAll("-", "_");
      stages.push({ id: `overlay_${name}`, label: `ovl ${name}` });
    }
    const overlays: Array<[boolean, string]> = [
      [draft.overlay.raw, "raw"],
      [draft.overlay.tracked, "tracked"],
      [draft.overlay.final, "final"],
      [draft.overlay.faces, "faces"],
    ];
    for (const [on, mode] of overlays) {
      if (on) {
        stages.push({ id: `overlay_${mode}`, label: `ovl ${mode}` });
      }
    }
  }
  return stages;
}

export type CellState = "waiting" | "active" | "done" | "failed";

export function stageStates(
  stages: PlannedStage[],
  job: JobSnapshot,
): CellState[] {
  const index = stages.findIndex((stage) => stage.id === job.stage);
  return stages.map((_, position) => {
    if (job.status === "completed") {
      return "done";
    }
    if (index === -1) {
      return "waiting";
    }
    if (position < index) {
      return "done";
    }
    if (position > index) {
      return "waiting";
    }
    if (job.status === "failed") {
      return "failed";
    }
    if (job.status === "running" || job.status === "cancelling") {
      return "active";
    }
    return job.status === "cancelled" ? "waiting" : "done";
  });
}

/** Coarse node state for the pipeline flow diagram. */
export type NodeState = "skipped" | "waiting" | "ready" | "active" | "done" | "failed";

export function groupState(
  prefix: "inference" | "postprocess" | "overlay",
  stages: PlannedStage[],
  states: CellState[],
): NodeState {
  const members = stages
    .map((stage, position) => ({ stage, state: states[position] }))
    .filter(({ stage }) => stage.id.startsWith(prefix));
  if (members.length === 0) {
    return "skipped";
  }
  if (members.some(({ state }) => state === "failed")) {
    return "failed";
  }
  if (members.some(({ state }) => state === "active")) {
    return "active";
  }
  if (members.every(({ state }) => state === "done")) {
    return "done";
  }
  return "waiting";
}
