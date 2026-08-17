import type {
  AppSettings,
  InferenceMode,
  OverlayExecutionMode,
  PipelineDraft,
  SettingsView,
} from "../../../shared/types";


export interface InspectorActions {
  inference: (values: Partial<PipelineDraft["inference"]>) => void;
  postprocess: (values: Partial<PipelineDraft["postprocess"]>) => void;
  overlay: (values: Partial<PipelineDraft["overlay"]>) => void;
  execution: (values: Partial<PipelineDraft["execution"]>) => void;
  settings: (values: Partial<AppSettings>) => void;
  pickSqlite: (target: "inference" | "tracked" | "final") => void;
  changeMode: (mode: InferenceMode) => void;
  changeModel: (model: PipelineDraft["inference"]["segmentationModel"]) => void;
  changeFaceModel: (model: PipelineDraft["inference"]["faceModel"]) => void;
  changeOverlayExecution: (mode: OverlayExecutionMode) => void;
  pickBackendRoot: () => void;
  pickPython: () => void;
}

export interface InspectorSectionProps {
  draft: PipelineDraft;
  settings: AppSettings;
  platform: NodeJS.Platform;
  busy: boolean;
  open: Record<string, boolean>;
  advanced: boolean;
  onToggle: (key: string) => void;
  actions: InspectorActions;
}

export type InspectorPanelProps = Omit<InspectorSectionProps, "advanced"> & {
  viewMode: SettingsView;
  onViewModeChange: (mode: SettingsView) => void;
};
