import { contextBridge, ipcRenderer, webUtils } from "electron";
import type {
  AppSettings,
  BootstrapData,
  FilePickerKind,
  HardwareMetrics,
  LivePreviewFrame,
  JobSnapshot,
  MaskStudioApi,
  PipelineDraft,
  VideoProbe,
} from "../shared/types";

const api: MaskStudioApi = {
  bootstrap: () => ipcRenderer.invoke("app:bootstrap") as Promise<BootstrapData>,
  pickFile: (kind: FilePickerKind) =>
    ipcRenderer.invoke("dialog:pick-file", kind) as Promise<string | null>,
  pickVideos: () =>
    ipcRenderer.invoke("dialog:pick-videos") as Promise<string[]>,
  pickDirectory: () =>
    ipcRenderer.invoke("dialog:pick-directory") as Promise<string | null>,
  probeVideo: (path: string, settings: AppSettings) =>
    ipcRenderer.invoke("video:probe", path, settings) as Promise<VideoProbe>,
  pathForFile: (file: File) => {
    try {
      return webUtils.getPathForFile(file) || null;
    } catch {
      return null;
    }
  },
  saveSettings: (settings: AppSettings) =>
    ipcRenderer.invoke(
      "settings:save",
      settings,
    ) as Promise<AppSettings>,
  validateWorkflow: (draft: PipelineDraft, settings: AppSettings) =>
    ipcRenderer.invoke(
      "workflow:validate",
      draft,
      settings,
    ) as Promise<JobSnapshot>,
  startWorkflow: (draft: PipelineDraft, settings: AppSettings) =>
    ipcRenderer.invoke(
      "workflow:start",
      draft,
      settings,
    ) as Promise<JobSnapshot>,
  cancelWorkflow: () =>
    ipcRenderer.invoke("workflow:cancel") as Promise<JobSnapshot>,
  setPreviewEnabled: (enabled: boolean) =>
    ipcRenderer.invoke("preview:set-enabled", enabled) as Promise<void>,
  sampleHardware: () =>
    ipcRenderer.invoke("system:sample-hardware") as Promise<HardwareMetrics>,
  openOutput: (path: string) =>
    ipcRenderer.invoke("shell:open-output", path) as Promise<string>,
  onJobUpdate: (callback: (job: JobSnapshot) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, job: JobSnapshot) =>
      callback(job);
    ipcRenderer.on("job:update", listener);
    return () => ipcRenderer.removeListener("job:update", listener);
  },
  onPreviewUpdate: (callback: (frame: LivePreviewFrame) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, frame: LivePreviewFrame) =>
      callback(frame);
    ipcRenderer.on("preview:update", listener);
    return () => ipcRenderer.removeListener("preview:update", listener);
  },
};

contextBridge.exposeInMainWorld("maskStudio", api);
