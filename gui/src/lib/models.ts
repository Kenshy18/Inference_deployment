import type {
  FaceBackend,
  FaceModel,
  PipelineDraft,
} from "../../shared/types";

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

export interface FaceModelSpec {
  id: FaceModel;
  label: string;
  note: string;
  backends: ReadonlyArray<{
    value: FaceBackend;
    label: string;
    id: string;
  }>;
}

const FAST = "TensorRT（高速・推奨）";
const SLOW = "PyTorch（互換）";

/** Mirrors InstanceSegmentation/inference/registry.py — a model may only be
 *  launched with a backend it registers, so the picker offers nothing else. */
export const SEGMENTATION_MODELS: readonly ModelSpec[] = [
  {
    id: "eva02_cascade",
    label: "EVA-02 + Cascade",
    note: "旧世代のEVA-02セグメンテーション",
    backends: [
      { value: "tensorrt-backbone", label: FAST, id: "tensorrt-backbone" },
      { value: "pytorch", label: SLOW, id: "pytorch" },
    ],
  },
  {
    id: "dinov3_cascade",
    label: "DINOv3 + Cascade",
    note: "DINOv3バックボーン + Cascade Mask R-CNN",
    backends: [
      { value: "tensorrt-backbone", label: FAST, id: "tensorrt-backbone" },
    ],
  },
  {
    id: "dinov3_codino",
    label: "Co-DINO（巨大）",
    note: "高精度の大型DINOv3 + Co-DINO",
    backends: [
      { value: "tensorrt-fast", label: FAST, id: "tensorrt-fast" },
      { value: "pytorch", label: SLOW, id: "pytorch" },
    ],
  },
  {
    id: "dinov3_codino_mh0",
    label: "Co-DINO（高速）",
    note: "高速・小型DINOv3 + Co-DINO（MH0）",
    backends: [
      { value: "tensorrt-fast", label: FAST, id: "tensorrt-fast" },
      { value: "pytorch", label: SLOW, id: "pytorch" },
    ],
  },
];

/** Mirrors InstanceSegmentation/inference/registry.py face registrations. */
export const FACE_MODELS: readonly FaceModelSpec[] = [
  {
    id: "face_dino_v2",
    label: "Face DINO v2（新）",
    note: "頭部box・顔楕円/マスク・キーポイント",
    backends: [
      { value: "tensorrt-fast", label: FAST, id: "tensorrt-fast" },
    ],
  },
  {
    id: "rtdetr_head_face",
    label: "RT-DETR（旧）",
    note: "Face / Head box",
    backends: [{ value: "pytorch", label: SLOW, id: "pytorch" }],
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

export function faceModelSpec(id: FaceModel): FaceModelSpec {
  return FACE_MODELS.find((model) => model.id === id) ?? FACE_MODELS[0];
}

export function defaultFaceBackend(id: FaceModel): FaceBackend {
  return faceModelSpec(id).backends[0].value;
}

/** Snap a stored backend onto the picked model; older drafts may hold "auto"
 *  or a backend the new model does not register. */
export function normalizeBackend(id: Model, backend: Backend): Backend {
  const spec = modelSpec(id);
  return spec.backends.some((option) => option.value === backend)
    ? backend
    : spec.backends[0].value;
}

export function normalizeFaceBackend(
  id: FaceModel,
  backend: FaceBackend,
): FaceBackend {
  const spec = faceModelSpec(id);
  return spec.backends.some((option) => option.value === backend)
    ? backend
    : spec.backends[0].value;
}
