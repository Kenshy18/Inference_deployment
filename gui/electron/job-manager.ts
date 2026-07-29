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
import { emptyTelemetry, parseTelemetryLine } from "./telemetry";

const MAX_LOG_LINES = 2_000;

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
    telemetry: { ...value.telemetry },
  };
}

export class JobManager extends EventEmitter {
  private readonly jobsRoot: string;
  private current: JobSnapshot = emptyJob();
  private child: ChildProcess | null = null;

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

    const id = new Date().toISOString().replaceAll(/[:.]/g, "-");
    const jobDir = path.join(this.jobsRoot, id);
    fs.mkdirSync(jobDir, { recursive: true });
    const configPath = path.join(jobDir, "orchestration.json");
    const generatedPolicy = buildClassPostprocessPolicy(draft);
    let effectiveDraft = draft;
    if (generatedPolicy !== null) {
      const policyPath = path.join(jobDir, "class_postprocess_policy.json");
      fs.writeFileSync(
        policyPath,
        `${JSON.stringify(generatedPolicy, null, 2)}\n`,
        "utf8",
      );
      effectiveDraft = {
        ...draft,
        postprocess: {
          ...draft.postprocess,
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
      outputRoot: draft.outputRoot,
      telemetry: emptyTelemetry(draft.inference.maxFrames),
    };
    this.emitUpdate();

    const launch = buildLaunchSpec(settings, configPath, dryRun);
    this.appendLog(`$ ${[launch.executable, ...launch.args].join(" ")}`);
    const child = spawn(launch.executable, launch.args, {
      cwd: launch.cwd,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONDONTWRITEBYTECODE: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
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
          this.current.artifacts = this.readArtifacts(draft.outputRoot);
        }
      }
      this.emitUpdate();
    };

    child.stdout.on("data", (chunk: Buffer) => this.consumeChunk(chunk));
    child.stderr.on("data", (chunk: Buffer) => this.consumeChunk(chunk));
    child.once("error", (error) => finalize(null, error));
    child.once("close", (code) => finalize(code));
    return this.snapshot();
  }

  cancel(): JobSnapshot {
    if (this.child === null) {
      return this.snapshot();
    }
    this.current.status = "cancelling";
    this.appendLog("キャンセル要求を送信しました。");
    this.child.kill("SIGTERM");
    this.emitUpdate();
    return this.snapshot();
  }

  private consumeChunk(chunk: Buffer): void {
    const text = chunk.toString("utf8");
    for (const line of text.split(/\r?\n/)) {
      if (!line) {
        continue;
      }
      const match = /^\[([^\]]+)]/.exec(line);
      if (match) {
        this.current.stage = match[1];
      }
      this.current.telemetry = parseTelemetryLine(
        this.current.telemetry,
        line,
      );
      this.appendLog(line);
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
