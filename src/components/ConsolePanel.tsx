import { useEffect, useMemo, useRef, useState } from "react";
import type { JobSnapshot } from "../../shared/types";
import { Check, Panel } from "./ui";

const TAG = /^(\[[^\]]+])\s?/;

function lineClass(line: string): string {
  if (line.startsWith("$ ")) {
    return "line line--cmd";
  }
  if (/error|traceback|exception|failed|not found/i.test(line)) {
    return "line line--err";
  }
  if (/warn/i.test(line)) {
    return "line line--warn";
  }
  return "line";
}

export function ConsolePanel({ job }: { job: JobSnapshot }) {
  const [filter, setFilter] = useState("");
  const [follow, setFollow] = useState(true);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  const lines = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return job.logs
      .map((text, index) => ({ text, index }))
      .filter(({ text }) =>
        needle ? text.toLowerCase().includes(needle) : true,
      );
  }, [filter, job.logs]);

  useEffect(() => {
    if (!follow || !bodyRef.current) {
      return;
    }
    bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [follow, lines]);

  return (
    <Panel
      title="Console"
      meta={job.stage ?? undefined}
      className="console"
      actions={
        <>
          <span className="panel__meta">{job.logs.length} lines</span>
          <label className="console__filter">
            <input
              value={filter}
              placeholder="フィルタ"
              spellCheck={false}
              onChange={(event) => setFilter(event.target.value)}
            />
          </label>
          <Check checked={follow} onChange={setFollow} label="追従" />
        </>
      }
    >
      <div className="console__body" ref={bodyRef}>
        {job.logs.length === 0 ? (
          <div className="empty">
            Dry Run または実行を開始すると、orchestration の出力がここに流れます。
          </div>
        ) : (
          lines.map(({ text, index }) => {
            const match = TAG.exec(text);
            return (
              <div className={lineClass(text)} key={index}>
                <i>{index + 1}</i>
                <code>
                  {match && <span className="line__tag">{match[1]} </span>}
                  {match ? text.slice(match[0].length) : text}
                </code>
              </div>
            );
          })
        )}
      </div>
    </Panel>
  );
}
