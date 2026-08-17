import type { SettingsView } from "../../shared/types";
import {
  InferenceSection,
  OverlaySection,
  PostprocessSection,
  RuntimeSection,
} from "./inspector";
import type { InspectorPanelProps } from "./inspector";
import { Panel, Segment } from "./ui";


export type { InspectorActions } from "./inspector";

export function InspectorPanel({
  draft,
  settings,
  platform,
  busy,
  open,
  viewMode,
  onViewModeChange,
  onToggle,
  actions,
}: InspectorPanelProps) {
  const advanced = viewMode === "advanced";
  const sectionProps = {
    draft,
    settings,
    platform,
    busy,
    open,
    advanced,
    onToggle,
    actions,
  };

  return (
    <Panel
      title="Inspector"
      className="panel--inspector"
      actions={
        <Segment<SettingsView>
          value={viewMode}
          disabled={busy}
          onChange={onViewModeChange}
          options={[
            { value: "simple", label: "簡単" },
            { value: "advanced", label: "詳細" },
          ]}
        />
      }
    >
      <div className="panel__body">
        <InferenceSection {...sectionProps} />
        <PostprocessSection {...sectionProps} />
        <OverlaySection {...sectionProps} />
        <RuntimeSection {...sectionProps} />
      </div>
    </Panel>
  );
}
