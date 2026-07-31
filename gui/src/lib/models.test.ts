import { describe, expect, it } from "vitest";
import {
  FACE_MODELS,
  SEGMENTATION_MODELS,
  defaultBackend,
  defaultFaceBackend,
  faceModelSpec,
  modelSpec,
  normalizeBackend,
  normalizeFaceBackend,
} from "./models";

describe("segmentation model catalog", () => {
  it("offers only the backends registry.py registers for each model", () => {
    expect(
      SEGMENTATION_MODELS.map((model) => [
        model.id,
        model.backends.map((backend) => backend.id),
      ]),
    ).toEqual([
      ["eva02_cascade", ["tensorrt-backbone", "pytorch"]],
      ["dinov3_cascade", ["tensorrt-backbone"]],
      ["dinov3_codino", ["tensorrt-fast", "pytorch"]],
      ["dinov3_codino_mh0", ["tensorrt-fast", "pytorch"]],
    ]);
  });

  it("uses the product-facing model version names", () => {
    expect(modelSpec("eva02_cascade").label).toBe("V1");
    expect(modelSpec("dinov3_cascade").label).toBe("V2");
    expect(modelSpec("dinov3_codino").label).toBe("V3");
    expect(modelSpec("dinov3_codino_mh0").label).toBe("v3-lite");
  });

  it("defaults every model to its fast backend", () => {
    expect(defaultBackend("eva02_cascade")).toBe("tensorrt-backbone");
    expect(defaultBackend("dinov3_cascade")).toBe("tensorrt-backbone");
    expect(defaultBackend("dinov3_codino")).toBe("tensorrt-fast");
    expect(defaultBackend("dinov3_codino_mh0")).toBe("tensorrt-fast");
    for (const model of SEGMENTATION_MODELS) {
      expect(model.backends[0].label).toBe("高速（デフォルト）");
    }
  });

  it("leaves V2 without a backend choice", () => {
    expect(modelSpec("dinov3_cascade").backends).toHaveLength(1);
  });

  it("snaps a stored backend the picked model cannot run", () => {
    expect(normalizeBackend("dinov3_cascade", "pytorch")).toBe(
      "tensorrt-backbone",
    );
    expect(normalizeBackend("dinov3_codino", "tensorrt-backbone")).toBe(
      "tensorrt-fast",
    );
    expect(normalizeBackend("eva02_cascade", "auto")).toBe(
      "tensorrt-backbone",
    );
    expect(normalizeBackend("eva02_cascade", "pytorch")).toBe("pytorch");
  });

  it("offers both current face detectors", () => {
    expect(FACE_MODELS.map((model) => model.id)).toEqual([
      "face_dino_v2",
      "rtdetr_head_face",
    ]);
    expect(faceModelSpec("rtdetr_head_face").label).toBe("Face V1");
    expect(faceModelSpec("face_dino_v2").label).toBe("Face V2");
  });

  it("shows and normalizes the engine supported by each face model", () => {
    expect(defaultFaceBackend("face_dino_v2")).toBe("tensorrt-fast");
    expect(defaultFaceBackend("rtdetr_head_face")).toBe("pytorch");
    expect(faceModelSpec("face_dino_v2").backends).toHaveLength(1);
    expect(faceModelSpec("face_dino_v2").backends[0].label).toBe(
      "高速（デフォルト）",
    );
    expect(faceModelSpec("rtdetr_head_face").backends[0].label).toBe(
      "低速（安定）",
    );
    expect(normalizeFaceBackend("rtdetr_head_face", "tensorrt-fast")).toBe(
      "pytorch",
    );
  });
});
