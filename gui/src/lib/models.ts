import type { PipelineDraft } from "../../shared/types";

type Model = PipelineDraft["inference"]["segmentationModel"];
type Backend = PipelineDraft["inference"]["segmentationBackend"];

export interface BackendOption {
  value: Backend;
  label: string;
  /** Backend id the inference CLI actually receives. */
  id: string;
}

export interface ModelSpec {
  id: Model;
  label: string;
  note: string;
  backends: BackendOption[];
}

const FAST = "Fast (Default)";
const SLOW = "Slow (Stable)";

/** Mirrors InstanceSegmentation/inference/registry.py — a model may only be
 *  launched with a backend it registers, so the picker offers nothing else. */
export const SEGMENTATION_MODELS: readonly ModelSpec[] = [
  {
    id: "eva02_cascade",
    label: "V1 (EVA)",
    note: "EVA-02 + Cascade",
    backends: [
      { value: "tensorrt-backbone", label: FAST, id: "tensorrt-backbone" },
      { value: "pytorch", label: SLOW, id: "pytorch" },
    ],
  },
  {
    id: "dinov3_cascade",
    label: "V2 (DINO)",
    note: "DINOv3 + Cascade",
    backends: [
      { value: "tensorrt-backbone", label: FAST, id: "tensorrt-backbone" },
    ],
  },
  {
    id: "dinov3_codino",
    label: "V3-heavy",
    note: "DINOv3 + Co-DINO",
    backends: [
      { value: "tensorrt-fast", label: FAST, id: "tensorrt-fast" },
      { value: "pytorch", label: SLOW, id: "pytorch" },
    ],
  },
];

export function modelSpec(id: Model): ModelSpec {
  return (
    SEGMENTATION_MODELS.find((model) => model.id === id) ??
    SEGMENTATION_MODELS[0]
  );
}

export function defaultBackend(id: Model): Backend {
  return modelSpec(id).backends[0].value;
}

/** Snap a stored backend onto the picked model; older drafts may hold "auto"
 *  or a backend the new model does not register. */
export function normalizeBackend(id: Model, backend: Backend): Backend {
  const spec = modelSpec(id);
  return spec.backends.some((option) => option.value === backend)
    ? backend
    : spec.backends[0].value;
}
