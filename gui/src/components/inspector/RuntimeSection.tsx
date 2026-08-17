import {
  Check,
  PathInput,
  Row,
  Section,
  Segment,
  TextInput,
} from "../ui";
import type { InspectorSectionProps } from "./types";


export function RuntimeSection({
  draft,
  settings,
  platform,
  busy,
  open,
  advanced,
  onToggle,
  actions,
}: InspectorSectionProps) {
  return (
        <Section
          name="実行環境"
          open={open.runtime}
          onToggle={() => onToggle("runtime")}
          badge={settings.backendMode === "wsl" ? "wsl2" : "native"}
        >
          {advanced && (
            <Row label="再開">
              <Check
                checked={draft.execution.resume}
                disabled={busy}
                onChange={(resume) => actions.execution({ resume })}
                label="完了済みstageを再利用"
              />
            </Row>
          )}
          <Row
            label="バックエンド"
            hint={platform === "win32" ? "Windows版はWSL2固定" : undefined}
          >
            <Segment
              value={settings.backendMode}
              disabled={busy || platform === "win32"}
              onChange={(backendMode) =>
                actions.settings({ backendMode })
              }
              options={[
                { value: "native", label: "Native" },
                { value: "wsl", label: "WSL2" },
              ]}
            />
          </Row>
          <Row label="リポジトリroot" stack>
            <PathInput
              value={settings.backendRoot}
              placeholder="/home/user/inference_backend2"
              disabled={busy}
              browseDisabled={settings.backendMode === "wsl"}
              onChange={(backendRoot) =>
                actions.settings({ backendRoot })
              }
              onBrowse={actions.pickBackendRoot}
            />
          </Row>
          <Row label="実行Python" stack>
            <PathInput
              value={settings.runtimePython}
              placeholder="/path/to/python3.10"
              disabled={busy}
              browseDisabled={settings.backendMode === "wsl"}
              onChange={(runtimePython) =>
                actions.settings({ runtimePython })
              }
              onBrowse={actions.pickPython}
            />
          </Row>
          {settings.backendMode === "wsl" && (
            <Row label="WSL distribution" hint="wsl.exe -l -v">
              <TextInput
                value={settings.wslDistro}
                placeholder="Ubuntu-24.04"
                disabled={busy}
                mono
                onChange={(wslDistro) =>
                  actions.settings({ wslDistro })
                }
              />
            </Row>
          )}
        </Section>
  );
}
