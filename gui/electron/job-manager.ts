import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import type {
  AppSettings,
  ArtifactMap,
  JobSnapshot,
  PipelineDraft,
} from "../shared/types";
import {
  buildClassPostprocessPolicy,
  buildLaunchSpec,
  buildOrchestrationConfig,
} from "./orchestration";
import { availableOutputRoot } from "./output-root";
import { emptyTelemetry, parseTelemetryLine } from "./telemetry";

const MAX_LOG_LINES = 2_000;
const LIVE_PREVIEW_MARKER = "[live-preview] ";

export interface LivePreviewFileEvent {
  jobId: string | null;
  path: string;
  phase: string;
  frameIndex: number;
  timestampSeconds: number;
  model: string;
  stage?: string;
  status?: string;
  detail?: string;
  width: number;
  height: number;
  generatedAtMs: number;
  dropped: number;
}

function emptyJob(): JobSnapshot {
  return {
    id: null,
    status: "idle",
    dryRun: false,
    stage: null,
    startedAt: null,
    completedAt: null,
    exitCode: null,
    error: null,
    logs: [],
    outputRoot: null,
    artifacts: {},
    telemetry: emptyTelemetry(),
  };
}

function copySnapshot(value: JobSnapshot): JobSnapshot {
  return {
    ...value,
    logs: [...value.logs],
    artifacts: { ...value.artifacts },
    telemetry: {
      ...value.telemetry,
      phases: {
        segmentation_inference: {
          ...value.telemetry.phases.segmentation_inference,
        },
        face_inference: { ...value.telemetry.phases.face_inference },
        postprocess: { ...value.telemetry.phases.postprocess },
        overlay: { ...value.telemetry.phases.overlay },
      },
    },
  };
}

export class JobManager extends EventEmitter {
  private readonly jobsRoot: string;
  private current: JobSnapshot = emptyJob();
  private child: ChildProcess | null = null;
  private stdoutBuffer = "";
  private stderrBuffer = "";
  private previewEnabled = false;
  private previewControl: string | null = null;

  constructor(jobsRoot: string) {
    super();
    this.jobsRoot = jobsRoot;
  }

  snapshot(): JobSnapshot {
    return copySnapshot(this.current);
  }

  async run(
    draft: PipelineDraft,
    settings: AppSettings,
    dryRun: boolean,
  ): Promise<JobSnapshot> {
    if (this.child !== null) {
      throw new Error("別のジョブが実行中です。");
    }
    if (!draft.inputVideo.trim()) {
      throw new Error("入力動画を選択してください。");
    }
    if (!draft.outputRoot.trim()) {
      throw new Error("出力フォルダを選択してください。");
    }
    if (!settings.backendRoot.trim() || !settings.runtimePython.trim()) {
      throw new Error("バックエンドとPythonのパスを設定してください。");
    }

    const outputRoot = availableOutputRoot(
      draft.outputRoot,
      draft.execution.resume,
    );
    const id = new Date().toISOString().replaceAll(/[:.]/g, "-");
    const jobDir = path.join(this.jobsRoot, id);
    fs.mkdirSync(jobDir, { recursive: true });
    this.previewControl = path.join(jobDir, "preview.enabled");
    const configPath = path.join(jobDir, "orchestration.json");
    const generatedPolicy = buildClassPostprocessPolicy(draft);
    let effectiveDraft = { ...draft, outputRoot };
    if (generatedPolicy !== null) {
      const policyPath = path.join(jobDir, "class_postprocess_policy.json");
      fs.writeFileSync(
        policyPath,
        `${JSON.stringify(generatedPolicy, null, 2)}\n`,
        "utf8",
      );
      effectiveDraft = {
        ...effectiveDraft,
        postprocess: {
          ...effectiveDraft.postprocess,
          classPostprocessPolicyJson: policyPath,
        },
      };
    }
    const config = buildOrchestrationConfig(effectiveDraft, settings);
    fs.writeFileSync(
      configPath,
      `${JSON.stringify(config, null, 2)}\n`,
      "utf8",
    );

    this.current = {
      ...emptyJob(),
      id,
      status: dryRun ? "validating" : "running",
      dryRun,
      startedAt: new Date().toISOString(),
      outputRoot,
      telemetry: emptyTelemetry(draft.inference.maxFrames),
    };
    this.syncPreviewControl();
    this.stdoutBuffer = "";
    this.stderrBuffer = "";
    this.emitUpdate();

    const launch = buildLaunchSpec(settings, configPath, dryRun);
    if (outputRoot !== draft.outputRoot) {
      this.appendLog(
        `[gui] 既存の出力を保護するため新しい保存先を使用します: ${outputRoot}`,
      );
    }
    this.appendLog(`$ ${[launch.executable, ...launch.args].join(" ")}`);
    const child = spawn(launch.executable, launch.args, {
      cwd: launch.cwd,
      env: {
        ...process.env,
        MASK_PIPELINE_PROGRESS_INTERVAL_SEC: "0.1",
        MASK_PIPELINE_PREVIEW_PATH: path.join(
          outputRoot,
          ".live-preview",
          "latest.jpg",
        ),
        MASK_PIPELINE_PREVIEW_CONTROL_PATH: this.previewControl,
        MASK_PIPELINE_PREVIEW_INTERVAL_FRAMES: "5",
        MASK_PIPELINE_PREVIEW_WIDTH: "960",
        MASK_PIPELINE_PREVIEW_HEIGHT: "540",
        MASK_PIPELINE_PREVIEW_JPEG_QUALITY: "85",
        MASK_PIPELINE_INFERENCE_PREVIEW_FPS: "5",
        MASK_PIPELINE_POSTPROCESS_PREVIEW_FPS: "5",
        PYTHONUNBUFFERED: "1",
        PYTHONDONTWRITEBYTECODE: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      detached: process.platform !== "win32",
    });
    this.child = child;

    let finalized = false;
    const finalize = (exitCode: number | null, error?: Error): void => {
      if (finalized) {
        return;
      }
      finalized = true;
      this.child = null;
      const wasCancelling = this.current.status === "cancelling";
      this.current.exitCode = exitCode;
      this.current.completedAt = new Date().toISOString();
      if (wasCancelling) {
        this.current.status = "cancelled";
      } else if (error || exitCode !== 0) {
        this.current.status = "failed";
        this.current.error =
          error?.message ?? `プロセスが終了コード ${exitCode} で終了しました。`;
      } else {
        this.current.status = dryRun ? "validated" : "completed";
        if (!dryRun) {
          this.current.artifacts = this.readArtifacts(outputRoot);
        }
      }
      this.emitUpdate();
    };

    child.stdout.on("data", (chunk: Buffer) =>
      this.consumeChunk(chunk, "stdout"),
    );
    child.stderr.on("data", (chunk: Buffer) =>
      this.consumeChunk(chunk, "stderr"),
    );
    child.once("error", (error) => finalize(null, error));
    child.once("close", (code) => {
      this.flushBufferedLines();
      finalize(code);
    });
    return this.snapshot();
  }

  cancel(): JobSnapshot {
    if (this.child === null) {
      return this.snapshot();
    }
    this.current.status = "cancelling";
    this.appendLog("キャンセル要求を送信しました。");
    const pid = this.child.pid;
    if (process.platform !== "win32" && pid !== undefined) {
      try {
        process.kill(-pid, "SIGTERM");
      } catch {
        this.child.kill("SIGTERM");
      }
    } else {
      this.child.kill("SIGTERM");
    }
    this.emitUpdate();
    return this.snapshot();
  }

  setPreviewEnabled(enabled: boolean): void {
    this.previewEnabled = enabled;
    this.syncPreviewControl();
  }

  private syncPreviewControl(): void {
    if (!this.previewControl) {
      return;
    }
    const control = this.previewControl;
    try {
      if (this.previewEnabled) {
        fs.mkdirSync(path.dirname(control), { recursive: true });
        fs.writeFileSync(control, "1\n", "utf8");
      } else if (fs.existsSync(control)) {
        fs.unlinkSync(control);
      }
    } catch {
      // Preview control is optional and must not affect the workflow.
    }
  }

  private consumeChunk(chunk: Buffer, source: "stdout" | "stderr"): void {
    const buffered =
      (source === "stdout" ? this.stdoutBuffer : this.stderrBuffer) +
      chunk.toString("utf8");
    const lines = buffered.split(/\r?\n/);
    const remainder = lines.pop() ?? "";
    if (source === "stdout") {
      this.stdoutBuffer = remainder;
    } else {
      this.stderrBuffer = remainder;
    }
    for (const line of lines) {
      this.consumeLine(line);
    }
  }

  private flushBufferedLines(): void {
    for (const line of [this.stdoutBuffer, this.stderrBuffer]) {
      if (line) {
        this.consumeLine(line);
      }
    }
    this.stdoutBuffer = "";
    this.stderrBuffer = "";
  }

  private consumeLine(line: string): void {
    if (!line) {
      return;
    }
    const previewMarker = line.indexOf(LIVE_PREVIEW_MARKER);
    if (previewMarker >= 0) {
      this.consumePreviewLine(
        line.slice(previewMarker + LIVE_PREVIEW_MARKER.length),
      );
      return;
    }
    const match = /^\[([^\]]+)]/.exec(line);
    if (match) {
      this.current.stage = match[1];
    }
    this.current.telemetry = parseTelemetryLine(
      this.current.telemetry,
      line,
    );
    if (!line.includes("[phase-progress]")) {
      this.appendLog(line);
    } else {
      this.emitUpdate();
    }
  }

  private consumePreviewLine(payloadText: string): void {
    try {
      const payload = JSON.parse(payloadText) as Record<string, unknown>;
      const previewPath = path.resolve(String(payload.path ?? ""));
      const outputRoot = this.current.outputRoot;
      if (!outputRoot) {
        return;
      }
      const root = path.resolve(outputRoot);
      if (
        previewPath !== root &&
        !previewPath.startsWith(`${root}${path.sep}`)
      ) {
        return;
      }
      this.emit("preview", {
        jobId: this.current.id,
        path: previewPath,
        phase: String(payload.phase ?? "inference"),
        frameIndex: Number(payload.frame_index ?? 0),
        timestampSeconds: Number(payload.timestamp_sec ?? 0),
        model: String(payload.model ?? ""),
        stage: payload.stage === undefined ? undefined : String(payload.stage),
        status: payload.status === undefined ? undefined : String(payload.status),
        detail: payload.detail === undefined ? undefined : String(payload.detail),
        width: Number(payload.width ?? 960),
        height: Number(payload.height ?? 540),
        generatedAtMs: Number(payload.generated_at_ms ?? Date.now()),
        dropped: Number(payload.dropped ?? 0),
      } satisfies LivePreviewFileEvent);
    } catch {
      // Live preview is optional and must never fail the workflow or pollute
      // its normal log stream.
    }
  }

  private appendLog(line: string): void {
    this.current.logs.push(line);
    if (this.current.logs.length > MAX_LOG_LINES) {
      this.current.logs.splice(0, this.current.logs.length - MAX_LOG_LINES);
    }
    this.emitUpdate();
  }

  private readArtifacts(outputRoot: string): ArtifactMap {
    try {
      const manifestPath = path.join(outputRoot, "run_manifest.json");
      const manifest = JSON.parse(
        fs.readFileSync(manifestPath, "utf8"),
      ) as { artifacts?: ArtifactMap };
      return manifest.artifacts ?? {};
    } catch {
      return {};
    }
  }

  private emitUpdate(): void {
    this.emit("update", this.snapshot());
  }
}
