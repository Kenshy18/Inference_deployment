import { describe, expect, it } from "vitest";
import { defaultDraft, emptyJob } from "../../lib/defaults";
import { buildMonitorFlow } from "./flow";

const queue = { total: 2, pending: 2, position: 0, activeTitle: null };

describe("buildMonitorFlow", () => {
  it("represents the default combined workflow in execution order", () => {
    const { nodes, overlayCount } = buildMonitorFlow(
      structuredClone(defaultDraft),
      structuredClone(emptyJob),
      queue,
    );

    expect(nodes.map((node) => node.key)).toEqual([
      "input",
      "segmentation",
      "face",
      "postprocess",
      "overlay",
      "output",
    ]);
    expect(overlayCount).toBe(2);
  });

  it("shows one explicit reuse node when inference is disabled", () => {
    const draft = structuredClone(defaultDraft);
    draft.inference.enabled = false;
    draft.postprocess.enabled = false;
    draft.overlay.enabled = false;

    const { nodes, overlayCount } = buildMonitorFlow(
      draft,
      structuredClone(emptyJob),
      queue,
    );

    expect(nodes.map((node) => node.key)).toEqual(["input", "reuse", "output"]);
    expect(overlayCount).toBe(2);
  });
});
