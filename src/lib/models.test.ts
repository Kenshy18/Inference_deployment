import { describe, expect, it } from "vitest";
import {
  SEGMENTATION_MODELS,
  defaultBackend,
  modelSpec,
  normalizeBackend,
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
    ]);
  });

  it("names models by pipeline version", () => {
    expect(modelSpec("eva02_cascade").label).toBe("V1 (EVA)");
    expect(modelSpec("dinov3_cascade").label).toBe("V2 (DINO)");
    expect(modelSpec("dinov3_codino").label).toBe("V3-heavy");
  });

  it("defaults every model to its fast backend", () => {
    expect(defaultBackend("eva02_cascade")).toBe("tensorrt-backbone");
    expect(defaultBackend("dinov3_cascade")).toBe("tensorrt-backbone");
    expect(defaultBackend("dinov3_codino")).toBe("tensorrt-fast");
    for (const model of SEGMENTATION_MODELS) {
      expect(model.backends[0].label).toBe("Fast (Default)");
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
});
