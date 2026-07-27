import path from "node:path";
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
  PipelineDraft,
} from "../shared/types";
import { JobManager } from "./job-manager";
import { readSettings, writeSettings } from "./settings";

let mainWindow: BrowserWindow | null = null;
let jobManager: JobManager;

function settingsPath(): string {
  return path.join(app.getPath("userData"), "settings.json");
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
    title: "動画処理",
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
    settings: readSettings(settingsPath()),
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

  ipcMain.handle("dialog:pick-directory", async () => {
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
      writeSettings(settingsPath(), settings),
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

  ipcMain.handle("shell:open-output", async (_event, targetPath: string) => {
    if (!targetPath) {
      return "出力パスがありません。";
    }
    return shell.openPath(path.resolve(targetPath));
  });
}

app.whenReady().then(() => {
  jobManager = new JobManager(path.join(app.getPath("userData"), "jobs"));
  jobManager.on("update", (job) => {
    mainWindow?.webContents.send("job:update", job);
  });
  registerIpc();
  createWindow();

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
