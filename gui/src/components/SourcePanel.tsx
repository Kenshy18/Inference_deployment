import { useCallback, useEffect, useState } from "react";
import type { QueueItem } from "../../shared/types";
import { duration } from "../lib/format";
import { FilmIcon, FolderIcon, PlusIcon } from "./Icons";
import { Panel, PathInput, Row } from "./ui";

const STATUS_LABELS: Record<QueueItem["status"], string> = {
  pending: "未処理",
  processing: "処理中",
  done: "処理済み",
  failed: "失敗",
};

export interface QueueActions {
  addFiles: (files: File[]) => void;
  addByPicker: () => void;
  remove: (id: string) => void;
  stopAndRemove: (id: string) => void;
  requeue: (id: string) => void;
  openOutput: (id: string) => void;
  setOutputRoot: (value: string) => void;
  pickOutput: () => void;
}

interface MenuState {
  x: number;
  y: number;
  item: QueueItem;
}

function QueueEntry({
  item,
  progress,
  onContextMenu,
}: {
  item: QueueItem;
  progress: number | null;
  onContextMenu: (event: React.MouseEvent, item: QueueItem) => void;
}) {
  const meta =
    item.status === "done"
      ? item.summary ?? "完了"
      : item.status === "failed"
        ? item.error ?? "処理に失敗しました"
        : item.summary && item.status === "processing"
          ? item.summary
          : duration(item.durationSeconds);
  return (
    <div
      className={`qitem is-${item.status}`}
      title={`${item.path}\n右クリックで操作`}
      onContextMenu={(event) => onContextMenu(event, item)}
    >
      <span className="qitem__thumb">
        {item.thumbnail ? (
          <img src={item.thumbnail} alt="" draggable={false} />
        ) : (
          <FilmIcon />
        )}
        {item.durationSeconds !== null && item.status === "pending" && (
          <em>{duration(item.durationSeconds)}</em>
        )}
      </span>
      <span className="qitem__main">
        <b>{item.title}</b>
        <span className="qitem__meta" title={meta}>
          {meta}
        </span>
      </span>
      <span className={`qbadge is-${item.status}`}>
        {STATUS_LABELS[item.status]}
      </span>
      {item.status === "processing" && (
        <span
          className={`qitem__bar ${progress === null ? "is-indeterminate" : ""}`}
          style={
            progress === null
              ? undefined
              : { width: `${Math.round(progress * 100)}%` }
          }
        />
      )}
    </div>
  );
}

export function SourcePanel({
  queue,
  outputRoot,
  busy,
  activeProgress,
  actions,
}: {
  queue: QueueItem[];
  outputRoot: string;
  busy: boolean;
  activeProgress: number | null;
  actions: QueueActions;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [menu, setMenu] = useState<MenuState | null>(null);

  const pending = queue.filter((item) => item.status === "pending").length;

  const openMenu = useCallback((event: React.MouseEvent, item: QueueItem) => {
    event.preventDefault();
    setMenu({ x: event.clientX, y: event.clientY, item });
  }, []);

  useEffect(() => {
    if (!menu) {
      return;
    }
    const close = () => setMenu(null);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenu(null);
      }
    };
    window.addEventListener("click", close);
    window.addEventListener("blur", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("blur", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  const menuAction = (run: () => void) => () => {
    run();
    setMenu(null);
  };

  return (
    <Panel title="Source" meta={`${queue.length}本`}>
      <div className="panel__body">
        <Row
          label="出力リポジトリ"
          stack
          hint={
            busy
              ? "処理中は変更できません"
              : "各動画は「リポジトリ/動画名」へ出力します"
          }
        >
          <PathInput
            value={outputRoot}
            placeholder="全ジョブ共通の出力先"
            disabled={busy}
            onChange={actions.setOutputRoot}
            onBrowse={actions.pickOutput}
          />
        </Row>

        <div className="subhead subhead--bar">
          入力キュー
          {pending > 0 && <i className="subhead__note">残り{pending}</i>}
          <button
            type="button"
            className="btn btn--quiet btn--sm"
            onClick={actions.addByPicker}
            title="動画ファイルを選択して追加"
          >
            <PlusIcon />
            追加
          </button>
        </div>

        <div
          className={`queue ${dragOver ? "is-dragover" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            actions.addFiles(Array.from(event.dataTransfer.files));
          }}
        >
          {queue.length === 0 ? (
            <div className="queue__empty">
              <FilmIcon />
              ここに動画をドラッグ
              <br />
              または「追加」で選択します
            </div>
          ) : (
            queue.map((item) => (
              <QueueEntry
                key={item.id}
                item={item}
                progress={item.status === "processing" ? activeProgress : null}
                onContextMenu={openMenu}
              />
            ))
          )}
        </div>
      </div>

      {menu && (
        <div
          className="ctxmenu"
          style={{ left: menu.x, top: menu.y }}
          onContextMenu={(event) => event.preventDefault()}
        >
          {menu.item.status === "done" && (
            <button
              type="button"
              onClick={menuAction(() => actions.openOutput(menu.item.id))}
            >
              <FolderIcon />
              出力フォルダを開く
            </button>
          )}
          {(menu.item.status === "done" || menu.item.status === "failed") && (
            <button
              type="button"
              onClick={menuAction(() => actions.requeue(menu.item.id))}
            >
              再処理（未処理に戻す）
            </button>
          )}
          {menu.item.status === "processing" ? (
            <button
              type="button"
              className="is-danger"
              onClick={menuAction(() => actions.stopAndRemove(menu.item.id))}
            >
              停止して削除
            </button>
          ) : (
            <button
              type="button"
              className="is-danger"
              onClick={menuAction(() => actions.remove(menu.item.id))}
            >
              キューから削除
            </button>
          )}
        </div>
      )}
    </Panel>
  );
}
