import type { JobSnapshot } from "../../shared/types";
import { filename } from "../lib/format";
import {
  CheckIcon,
  FilmIcon,
  PlayIcon,
  StopIcon,
} from "./Icons";

export function TopBar({
  queueTotal,
  queuePending,
  outputRoot,
  job,
  busy,
  canRun,
  onRun,
  onCancel,
}: {
  queueTotal: number;
  queuePending: number;
  outputRoot: string;
  job: JobSnapshot;
  busy: boolean;
  canRun: boolean;
  onRun: (dryRun: boolean) => void;
  onCancel: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar__mark">
        <span className="topbar__logo">
          <FilmIcon />
        </span>
      </div>

      <div className="topbar__file">
        <span className="fchip" title="入力キューの状態">
          <i>QUEUE</i>
          <b className={queueTotal === 0 ? "is-empty" : ""}>
            {queueTotal === 0
              ? "キューは空"
              : `${queueTotal}本 · 残り${queuePending}`}
          </b>
        </span>
        <span className="fchip" title={outputRoot}>
          <i>OUT</i>
          <b className={outputRoot ? "" : "is-empty"}>
            {outputRoot ? filename(outputRoot) : "保存先未選択"}
          </b>
        </span>
      </div>

      <div className="topbar__transport">
        <button
          type="button"
          className="btn"
          disabled={!canRun}
          onClick={() => onRun(true)}
          title="設定と入力だけを検証 (Ctrl+D)"
        >
          <CheckIcon />
          Dry Run
          <kbd>^D</kbd>
        </button>
        {busy ? (
          <button
            type="button"
            className="btn btn--danger"
            disabled={job.status === "cancelling"}
            onClick={onCancel}
            title="キューの処理を停止 (Esc)"
          >
            <StopIcon />
            停止
            <kbd>Esc</kbd>
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--primary"
            disabled={!canRun}
            onClick={() => onRun(false)}
            title="キューを順番に処理 (Ctrl+Enter)"
          >
            <PlayIcon />
            実行
            <kbd>^↵</kbd>
          </button>
        )}
      </div>
    </header>
  );
}
