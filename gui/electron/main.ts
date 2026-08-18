import fs from "node:fs";
import path from "node:path";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  shell,
  type OpenDialogOptions,
} from "electron";
import type {
  AppSettings,
  FilePickerKind,
  LivePreviewFrame,
  PipelineDraft,
} from "../shared/types";
import { JobManager, type LivePreviewFileEvent } from "./job-manager";
import { HardwareSampler } from "./hardware";
import { probeVideo } from "./probe";
import { parseRuntimeOptions } from "./runtime-options";
import { runQaE2e } from "./qa-e2e";
import { readSettings, writeSettings } from "./settings";
import { runtimePathToHost } from "./wsl-bridge";
import {
  bindDeploymentSettings,
  loadDeploymentProfile,
} from "./deployment-profile";

let mainWindow: BrowserWindow | null = null;
let jobManager: JobManager;
const runtimeOptions = parseRuntimeOptions(process.argv, process.env);
const deploymentProfile = loadDeploymentProfile(
  process.argv,
  process.env,
  process.execPath,
);
if (deploymentProfile) {
  app.setPath("userData", deploymentProfile.userDataPath);
}
const execFileAsync = promisify(execFile);
const hardwareSampler = new HardwareSampler();
let pendingPreview: LivePreviewFileEvent | null = null;
let previewReadActive = false;

function queuePreview(frame: LivePreviewFileEvent): void {
  pendingPreview = frame;
  if (previewReadActive) {
    return;
  }
  previewReadActive = true;
  void (async () => {
    while (pendingPreview !== null) {
      const current = pendingPreview;
      pendingPreview = null;
      try {
        const bytes = await fs.promises.readFile(current.path);
        const output: LivePreviewFrame = {
          jobId: current.jobId,
          dataUrl: `data:image/jpeg;base64,${bytes.toString("base64")}`,
          phase: current.phase,
          frameIndex: current.frameIndex,
          timestampSeconds: current.timestampSeconds,
          model: current.model,
          stage: current.stage,
          status: current.status,
          detail: current.detail,
          width: current.width,
          height: current.height,
          generatedAtMs: current.generatedAtMs,
          dropped: current.dropped,
        };
        mainWindow?.webContents.send("preview:update", output);
      } catch {
        // A newer ring-buffer slot may replace a preview before it is read.
      }
    }
    previewReadActive = false;
  })();
}

async function openOutputFolder(targetPath: string): Promise<string> {
  const settings = readAppSettings();
  const hostPath = runtimePathToHost(targetPath, settings);
  const resolved = path.resolve(hostPath);
  if (process.env.MASK_STUDIO_AUTOMATION_NO_EXTERNAL === "1") {
    console.log(`[gui] automation open-output: ${resolved}`);
    return "";
  }
  if (process.platform === "linux" && process.env.WSL_DISTRO_NAME) {
    try {
      const { stdout } = await execFileAsync("wslpath", ["-w", resolved]);
      const windowsPath = stdout.trim();
      if (!windowsPath) {
        return `Windowsパスへ変換できませんでした: ${resolved}`;
      }
      await new Promise<void>((resolve, reject) => {
        const explorer = spawn("explorer.exe", [windowsPath], {
          detached: true,
          stdio: "ignore",
          windowsHide: false,
        });
        explorer.once("error", reject);
        explorer.once("spawn", () => {
          explorer.unref();
          resolve();
        });
      });
      return "";
    } catch (error) {
      return error instanceof Error
        ? `エクスプローラーを開けませんでした: ${error.message}`
        : "エクスプローラーを開けませんでした。";
    }
  }
  return shell.openPath(resolved);
}

if (runtimeOptions.softwareRendering) {
  // WSLg occasionally fails to create Chromium's GPU command buffer. The
  // pipeline child processes still use CUDA; this affects only GUI painting.
  app.disableHardwareAcceleration();
}
if (runtimeOptions.automationPort !== null) {
  app.commandLine.appendSwitch(
    "remote-debugging-address",
    runtimeOptions.automationAddress,
  );
  app.commandLine.appendSwitch(
    "remote-debugging-port",
    String(runtimeOptions.automationPort),
  );
}

function settingsPath(): string {
  return path.join(app.getPath("userData"), "settings.json");
}

function readAppSettings(): AppSettings {
  return bindDeploymentSettings(readSettings(settingsPath()), deploymentProfile);
}

function writeAppSettings(settings: AppSettings): AppSettings {
  return writeSettings(
    settingsPath(),
    bindDeploymentSettings(settings, deploymentProfile),
  );
}

function filePickerOptions(kind: FilePickerKind): OpenDialogOptions {
  if (kind === "video") {
    return {
      properties: ["openFile"],
      filters: [
        {
          name: "動画",
          extensions: ["mp4", "mov", "mkv", "avi", "webm"],
        },
        { name: "すべてのファイル", extensions: ["*"] },
      ],
    };
  }
  if (kind === "sqlite") {
    return {
      properties: ["openFile"],
      filters: [
        { name: "SQLite", extensions: ["sqlite", "db"] },
        { name: "すべてのファイル", extensions: ["*"] },
      ],
    };
  }
  return {
    properties: ["openFile"],
    filters:
      process.platform === "win32"
        ? [
            { name: "実行ファイル", extensions: ["exe"] },
            { name: "すべてのファイル", extensions: ["*"] },
          ]
        : [{ name: "すべてのファイル", extensions: ["*"] }],
  };
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: "#111317",
    title: "Mask Pipeline Studio",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  const developmentUrl = process.env.VITE_DEV_SERVER_URL;
  if (developmentUrl) {
    void mainWindow.loadURL(developmentUrl);
  } else {
    void mainWindow.loadFile(
      path.join(__dirname, "../../renderer/index.html"),
    );
  }
}

function registerIpc(): void {
  ipcMain.handle("app:bootstrap", () => ({
    platform: process.platform,
    settings: readAppSettings(),
    job: jobManager.snapshot(),
  }));

  ipcMain.handle(
    "dialog:pick-file",
    async (_event, kind: FilePickerKind) => {
      const options = filePickerOptions(kind);
      const result = mainWindow
        ? await dialog.showOpenDialog(mainWindow, options)
        : await dialog.showOpenDialog(options);
      return result.canceled ? null : result.filePaths[0] ?? null;
    },
  );

  ipcMain.handle("dialog:pick-videos", async () => {
    if (runtimeOptions.automationVideos.length > 0) {
      return runtimeOptions.automationVideos;
    }
    const options: OpenDialogOptions = {
      ...filePickerOptions("video"),
      properties: ["openFile", "multiSelections"],
    };
    const result = mainWindow
      ? await dialog.showOpenDialog(mainWindow, options)
      : await dialog.showOpenDialog(options);
    return result.canceled ? [] : result.filePaths;
  });

  ipcMain.handle(
    "video:probe",
    (_event, videoPath: string, settings: AppSettings) =>
      probeVideo(videoPath, settings),
  );

  ipcMain.handle("dialog:pick-directory", async () => {
    if (runtimeOptions.automationOutput) {
      return runtimeOptions.automationOutput;
    }
    const options: OpenDialogOptions = {
      properties: ["openDirectory", "createDirectory"],
    };
    const result = mainWindow
      ? await dialog.showOpenDialog(mainWindow, options)
      : await dialog.showOpenDialog(options);
    return result.canceled ? null : result.filePaths[0] ?? null;
  });

  ipcMain.handle(
    "settings:save",
    (_event, settings: AppSettings) =>
      writeAppSettings(settings),
  );

  ipcMain.handle(
    "workflow:validate",
    (_event, draft: PipelineDraft, settings: AppSettings) =>
      jobManager.run(draft, settings, true),
  );

  ipcMain.handle(
    "workflow:start",
    (_event, draft: PipelineDraft, settings: AppSettings) =>
      jobManager.run(draft, settings, false),
  );

  ipcMain.handle("workflow:cancel", () => jobManager.cancel());

  ipcMain.handle("preview:set-enabled", (_event, enabled: boolean) => {
    jobManager.setPreviewEnabled(Boolean(enabled));
  });

  ipcMain.handle("system:sample-hardware", () => hardwareSampler.sample());

  ipcMain.handle("shell:open-output", async (_event, targetPath: string) => {
    if (!targetPath) {
      return "出力パスがありません。";
    }
    return openOutputFolder(targetPath);
  });
}

app.whenReady().then(() => {
  console.log(
    `[gui] software-rendering=${runtimeOptions.softwareRendering} ` +
      `automation-port=${runtimeOptions.automationPort ?? "off"} ` +
      `release=${deploymentProfile?.releaseId ?? "development"}`,
  );
  jobManager = new JobManager(path.join(app.getPath("userData"), "jobs"));
  jobManager.on("update", (job) => {
    mainWindow?.webContents.send("job:update", job);
  });
  jobManager.on("preview", queuePreview);
  registerIpc();
  createWindow();
  if (
    mainWindow &&
    runtimeOptions.qaE2eInput &&
    runtimeOptions.qaE2eOutput &&
    runtimeOptions.qaE2eReport
  ) {
    void runQaE2e(mainWindow, {
      input: runtimeOptions.qaE2eInput,
      output: runtimeOptions.qaE2eOutput,
      report: runtimeOptions.qaE2eReport,
      maxFrames: runtimeOptions.qaE2eMaxFrames,
      cancelAfterMs: runtimeOptions.qaE2eCancelAfterMs,
    }).then(
      (passed) => app.exit(passed ? 0 : 1),
      () => app.exit(1),
    );
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  jobManager?.shutdown();
  hardwareSampler.close();
});
