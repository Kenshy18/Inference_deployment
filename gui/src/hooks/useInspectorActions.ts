import { useCallback, useMemo } from "react";
import type { Dispatch, SetStateAction } from "react";
import type {
  AppSettings,
  InferenceMode,
  OverlayExecutionMode,
  PipelineDraft,
} from "../../shared/types";
import type { InspectorActions } from "../components/InspectorPanel";
import { desktopApi } from "../lib/api";
import { defaultBackend, defaultFaceBackend } from "../lib/models";

interface InspectorActionState {
  draft: PipelineDraft;
  setDraft: Dispatch<SetStateAction<PipelineDraft>>;
  setSettings: Dispatch<SetStateAction<AppSettings>>;
}

/** Own all editor mutations and file-picker IPC used by the inspector. */
export function useInspectorActions({
  draft,
  setDraft,
  setSettings,
}: InspectorActionState): InspectorActions {
  const patchInference = useCallback(
    (values: Partial<PipelineDraft["inference"]>) =>
      setDraft((current) => ({
        ...current,
        inference: { ...current.inference, ...values },
      })),
    [setDraft],
  );

  const patchPostprocess = useCallback(
    (values: Partial<PipelineDraft["postprocess"]>) =>
      setDraft((current) => ({
        ...current,
        postprocess: { ...current.postprocess, ...values },
      })),
    [setDraft],
  );

  const patchOverlay = useCallback(
    (values: Partial<PipelineDraft["overlay"]>) =>
      setDraft((current) => ({
        ...current,
        overlay: { ...current.overlay, ...values },
      })),
    [setDraft],
  );

  const patchExecution = useCallback(
    (values: Partial<PipelineDraft["execution"]>) =>
      setDraft((current) => ({
        ...current,
        execution: { ...current.execution, ...values },
      })),
    [setDraft],
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

  return useMemo(
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
          parallelModels: false,
          parallelModelStaggerSeconds: 0,
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
          parallelModels: false,
          parallelModelStaggerSeconds: 0,
        }),
      changeFaceModel: (faceModel) => {
        patchInference({
          faceModel,
          faceBackend: defaultFaceBackend(faceModel),
          faceTrtBundle:
            faceModel === "face_dino_v2" ? draft.inference.faceTrtBundle : "",
          parallelModels: false,
          parallelModelStaggerSeconds: 0,
        });
        if (faceModel !== "face_dino_v2") {
          patchPostprocess({
            faceMaskTarget: "none",
            precomputeCutsDuringInference: draft.postprocess.cutDetect,
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
      setSettings,
    ],
  );
}
