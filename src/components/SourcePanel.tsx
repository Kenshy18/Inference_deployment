import type { JobSnapshot, PipelineDraft } from "../../shared/types";
import { filename } from "../lib/format";
import { FolderIcon } from "./Icons";
import { Panel, PathInput, Row, SubHead } from "./ui";

export interface SourceActions {
  setInputVideo: (value: string) => void;
  setOutputRoot: (value: string) => void;
  setInferenceSqlite: (value: string) => void;
  setTrackedSqlite: (value: string) => void;
  setFinalSqlite: (value: string) => void;
  pickVideo: () => void;
  pickOutput: () => void;
  pickSqlite: (target: "inference" | "tracked" | "final") => void;
  openOutput: () => void;
}

export function SourcePanel({
  draft,
  job,
  busy,
  actions,
}: {
  draft: PipelineDraft;
  job: JobSnapshot;
  busy: boolean;
  actions: SourceActions;
}) {
  const artifacts = Object.entries(job.artifacts);
  const reuseInference = !draft.inference.enabled;
  const reusePost = !draft.postprocess.enabled;

  return (
    <Panel title="Source">
      <div className="panel__body">
        <Row label="入力動画" stack>
          <PathInput
            value={draft.inputVideo}
            placeholder="mp4 / mov / mkv / avi"
            disabled={busy}
            onChange={actions.setInputVideo}
            onBrowse={actions.pickVideo}
          />
        </Row>
        <Row label="出力フォルダ" stack hint="空のフォルダか未作成のパス">
          <PathInput
            value={draft.outputRoot}
            placeholder="ジョブ成果物の保存先"
            disabled={busy}
            onChange={actions.setOutputRoot}
            onBrowse={actions.pickOutput}
          />
        </Row>

        <SubHead>既存データの再利用</SubHead>
        <Row
          label="inference SQLite"
          stack
          off={!reuseInference}
          hint={reuseInference ? undefined : "推論を無効にすると使用します"}
        >
          <PathInput
            value={draft.inference.inputSqlite}
            placeholder="inference.sqlite"
            disabled={busy || !reuseInference}
            onChange={actions.setInferenceSqlite}
            onBrowse={() => actions.pickSqlite("inference")}
          />
        </Row>
        <Row
          label="tracked SQLite"
          stack
          off={!reusePost}
          hint={reusePost ? undefined : "後処理を無効にすると使用します"}
        >
          <PathInput
            value={draft.postprocess.trackedSqlite}
            placeholder="tracked.sqlite"
            disabled={busy || !reusePost}
            onChange={actions.setTrackedSqlite}
            onBrowse={() => actions.pickSqlite("tracked")}
          />
        </Row>
        <Row label="final SQLite" stack off={!reusePost}>
          <PathInput
            value={draft.postprocess.finalSqlite}
            placeholder="predictions.sqlite"
            disabled={busy || !reusePost}
            onChange={actions.setFinalSqlite}
            onBrowse={() => actions.pickSqlite("final")}
          />
        </Row>

        <div className="artifacts">
          <div className="subhead" style={{ display: "flex", alignItems: "center" }}>
            成果物
            <span style={{ flex: 1 }} />
            <button
              type="button"
              className="btn btn--quiet btn--sm"
              onClick={actions.openOutput}
              disabled={!job.outputRoot && !draft.outputRoot}
              title="出力フォルダを開く"
            >
              <FolderIcon />
              開く
            </button>
          </div>
          {artifacts.length === 0 ? (
            <div className="empty">
              ジョブが完了すると
              <br />
              run_manifest.json の成果物を表示します
            </div>
          ) : (
            artifacts.map(([name, path]) => (
              <div className="artifact" key={name} title={path}>
                <b>{name.replaceAll("_", " ")}</b>
                <span>{filename(path)}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </Panel>
  );
}
