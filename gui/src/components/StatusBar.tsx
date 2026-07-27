import type { AppSettings, JobSnapshot } from "../../shared/types";
import { clock, duration, filename } from "../lib/format";

export function StatusBar({
  job,
  settings,
  statusLabel,
  elapsedSeconds,
  isElectron,
}: {
  job: JobSnapshot;
  settings: AppSettings;
  statusLabel: string;
  elapsedSeconds: number;
  isElectron: boolean;
}) {
  return (
    <footer className="status">
      <div className="status__item">
        <span className={`dot s-${job.status}`} />
        <b>{statusLabel}</b>
      </div>
      <div className="status__item">
        stage <b>{job.stage ?? "—"}</b>
      </div>
      <div className="status__item">
        経過 <b>{duration(elapsedSeconds)}</b>
      </div>
      <div className="status__item">
        開始 <b>{clock(job.startedAt)}</b>
      </div>
      <div className="status__item">
        終了 <b>{clock(job.completedAt)}</b>
      </div>
      <div className="status__item">
        exit <b>{job.exitCode ?? "—"}</b>
      </div>
      {job.error && (
        <div className="status__item is-path" title={job.error}>
          <b style={{ color: "var(--err)", direction: "ltr" }}>{job.error}</b>
        </div>
      )}

      <div className="status__item is-right">
        <b>{settings.backendMode === "wsl" ? `WSL2 ${settings.wslDistro}` : "NATIVE"}</b>
      </div>
      <div className="status__item is-path" title={settings.runtimePython}>
        python <b>{filename(settings.runtimePython)}</b>
      </div>
      <div className="status__item is-path" title={settings.backendRoot}>
        <b>{settings.backendRoot}</b>
      </div>
      {!isElectron && (
        <div className="status__item">
          <b style={{ color: "var(--run)" }}>BROWSER PREVIEW</b>
        </div>
      )}
    </footer>
  );
}
