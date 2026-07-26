import type { AppSettings, JobSnapshot, PipelineDraft } from "../../shared/types";
import { filename } from "../lib/format";
import {
  CheckIcon,
  CpuIcon,
  FilmIcon,
  PlayIcon,
  StopIcon,
} from "./Icons";

export function TopBar({
  draft,
  job,
  settings,
  busy,
  canRun,
  onRun,
  onCancel,
  onRuntime,
}: {
  draft: PipelineDraft;
  job: JobSnapshot;
  settings: AppSettings;
  busy: boolean;
  canRun: boolean;
  onRun: (dryRun: boolean) => void;
  onCancel: () => void;
  onRuntime: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar__mark">
        <FilmIcon />
        <span>Mask Pipeline</span>
      </div>

      <div className="topbar__file">
        <i>SRC</i>
        <b className={draft.inputVideo ? "" : "is-empty"} title={draft.inputVideo}>
          {draft.inputVideo ? filename(draft.inputVideo) : "動画未選択"}
        </b>
        <i>OUT</i>
        <b className={draft.outputRoot ? "" : "is-empty"} title={draft.outputRoot}>
          {draft.outputRoot ? filename(draft.outputRoot) : "保存先未選択"}
        </b>
      </div>

      <div className="topbar__transport">
        <button
          type="button"
          className="btn btn--quiet"
          onClick={onRuntime}
          title="実行環境の設定"
        >
          <CpuIcon />
          {settings.backendMode === "wsl" ? `WSL2 · ${settings.wslDistro}` : "Native"}
        </button>
        <span style={{ width: 8 }} />
        <button
          type="button"
          className="btn"
          disabled={!canRun}
          onClick={() => onRun(true)}
          title="設定と入力だけを検証 (Ctrl+D)"
        >
          <CheckIcon />
          Dry Run
        </button>
        <span style={{ width: 4 }} />
        {busy ? (
          <button
            type="button"
            className="btn btn--danger"
            disabled={job.status === "cancelling"}
            onClick={onCancel}
            title="ジョブを停止 (Esc)"
          >
            <StopIcon />
            停止
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--primary"
            disabled={!canRun}
            onClick={() => onRun(false)}
            title="ジョブを実行 (Ctrl+Enter)"
          >
            <PlayIcon />
            実行
          </button>
        )}
      </div>
    </header>
  );
}
