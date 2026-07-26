import type {
  AppSettings,
  InferenceMode,
  PipelineDraft,
} from "../../shared/types";
import {
  Check,
  NumberInput,
  Panel,
  PathInput,
  Row,
  Section,
  Segment,
  Select,
  Slider,
  TextInput,
} from "./ui";

type Draft = PipelineDraft;

export interface InspectorActions {
  inference: (values: Partial<Draft["inference"]>) => void;
  postprocess: (values: Partial<Draft["postprocess"]>) => void;
  overlay: (values: Partial<Draft["overlay"]>) => void;
  settings: (values: Partial<AppSettings>) => void;
  changeMode: (mode: InferenceMode) => void;
  changeModel: (model: Draft["inference"]["segmentationModel"]) => void;
  pickBackendRoot: () => void;
  pickPython: () => void;
}

export function InspectorPanel({
  draft,
  settings,
  busy,
  open,
  onToggle,
  actions,
}: {
  draft: Draft;
  settings: AppSettings;
  busy: boolean;
  open: Record<string, boolean>;
  onToggle: (key: string) => void;
  actions: InspectorActions;
}) {
  const { inference, postprocess, overlay } = draft;
  const faceOnly = inference.mode === "face";
  const usesFaces = inference.mode !== "segmentation";

  return (
    <Panel title="Inspector">
      <div className="panel__body">
        <Section
          name="Inference"
          open={open.inference}
          onToggle={() => onToggle("inference")}
          badge={inference.enabled ? "on" : "reuse"}
          badgeState={inference.enabled ? "on" : "off"}
        >
          <Row label="推論" always>
            <Check
              checked={inference.enabled}
              disabled={busy}
              onChange={(enabled) => actions.inference({ enabled })}
              label="GPUで新規に実行"
            />
          </Row>
          <Row label="処理モード" always>
            <Segment<InferenceMode>
              value={inference.mode}
              disabled={busy}
              onChange={actions.changeMode}
              options={[
                { value: "segmentation", label: "マスク" },
                { value: "segmentation-face", label: "マスク+顔" },
                { value: "face", label: "顔のみ" },
              ]}
            />
          </Row>
          {!faceOnly && (
            <>
              <Row label="モデル">
                <Select
                  value={inference.segmentationModel}
                  disabled={busy || !inference.enabled}
                  onChange={actions.changeModel}
                  options={[
                    { value: "dinov3_codino", label: "DINOv3 Co-DINO" },
                    { value: "eva02_cascade", label: "EVA-02 Cascade" },
                    { value: "dinov3_cascade", label: "DINOv3 Cascade" },
                  ]}
                />
              </Row>
              <Row label="バックエンド">
                <Select
                  value={inference.segmentationBackend}
                  disabled={busy || !inference.enabled}
                  onChange={(segmentationBackend) =>
                    actions.inference({ segmentationBackend })
                  }
                  options={[
                    { value: "auto", label: "Auto" },
                    { value: "tensorrt-fast", label: "TensorRT fast" },
                    { value: "tensorrt-backbone", label: "TensorRT backbone" },
                    { value: "pytorch", label: "PyTorch" },
                  ]}
                />
              </Row>
            </>
          )}
          {usesFaces && (
            <Row label="顔クラス">
              <TextInput
                value={inference.faceClasses.join(", ")}
                disabled={busy || !inference.enabled}
                mono
                onChange={(value) =>
                  actions.inference({
                    faceClasses: value
                      .split(",")
                      .map((item) => item.trim())
                      .filter(Boolean),
                  })
                }
              />
            </Row>
          )}
          <Row label="デバイス">
            <TextInput
              value={inference.device}
              disabled={busy || !inference.enabled}
              mono
              onChange={(device) => actions.inference({ device })}
            />
          </Row>
          <Row label="最大フレーム">
            <NumberInput
              value={inference.maxFrames}
              min={1}
              placeholder="全フレーム"
              disabled={busy || !inference.enabled}
              onChange={(maxFrames) => actions.inference({ maxFrames })}
            />
          </Row>
          <Row label="ウォームアップ">
            <NumberInput
              value={inference.warmupFrames}
              min={0}
              unit="f"
              disabled={busy || !inference.enabled}
              onChange={(value) =>
                actions.inference({ warmupFrames: value ?? 0 })
              }
            />
          </Row>
          {usesFaces && (
            <Row label="顔ウォームアップ">
              <NumberInput
                value={inference.faceWarmupIterations}
                min={0}
                unit="it"
                disabled={busy || !inference.enabled}
                onChange={(value) =>
                  actions.inference({ faceWarmupIterations: value ?? 0 })
                }
              />
            </Row>
          )}
          <Row label="SQLite">
            <Check
              checked={inference.fastSqlite}
              disabled={busy || !inference.enabled}
              onChange={(fastSqlite) => actions.inference({ fastSqlite })}
              label="高速書き込み"
              title="fast_sqlite: 書き込みを優先し耐障害性を下げます"
            />
          </Row>
        </Section>

        <Section
          name="Postprocess"
          open={open.postprocess}
          onToggle={() => onToggle("postprocess")}
          badge={postprocess.enabled ? "on" : "skip"}
          badgeState={postprocess.enabled ? "on" : "off"}
          muted={!postprocess.enabled}
        >
          <Row label="後処理" always>
            <Check
              checked={postprocess.enabled}
              disabled={busy || faceOnly}
              onChange={(enabled) => actions.postprocess({ enabled })}
              label={faceOnly ? "顔のみでは利用不可" : "マスクを整形"}
            />
          </Row>
          <Row label="形状">
            <Segment
              value={postprocess.shapeMode}
              disabled={busy}
              onChange={(shapeMode) => actions.postprocess({ shapeMode })}
              options={[
                { value: "polygon", label: "Polygon" },
                { value: "ellipse", label: "Ellipse" },
              ]}
            />
          </Row>
          <Row label="最小スコア">
            <Slider
              value={postprocess.scoreMin}
              min={0}
              max={1}
              step={0.01}
              disabled={busy}
              onChange={(scoreMin) => actions.postprocess({ scoreMin })}
            />
          </Row>
          <Row label="カット検出">
            <Check
              checked={postprocess.cutDetect}
              disabled={busy}
              onChange={(cutDetect) => actions.postprocess({ cutDetect })}
              label="シーン境界で track を分割"
            />
          </Row>
          <Row label="検出手法">
            <TextInput
              value={postprocess.cutMethod}
              disabled={busy || !postprocess.cutDetect}
              mono
              onChange={(cutMethod) => actions.postprocess({ cutMethod })}
            />
          </Row>
          <Row label="短命track除去">
            <NumberInput
              value={postprocess.removeShortTracksMaxFrames}
              min={0}
              unit="f"
              disabled={busy}
              onChange={(value) =>
                actions.postprocess({ removeShortTracksMaxFrames: value ?? 0 })
              }
            />
          </Row>
          <Row label="キーフレーム">
            <NumberInput
              value={postprocess.keyframeInterval}
              min={1}
              unit="f"
              disabled={busy}
              onChange={(value) =>
                actions.postprocess({ keyframeInterval: value ?? 1 })
              }
            />
          </Row>
          <Row label="デバイス" hint="orchestration は CPU 固定">
            <TextInput value="cpu" disabled mono onChange={() => undefined} />
          </Row>
        </Section>

        <Section
          name="Overlay"
          open={open.overlay}
          onToggle={() => onToggle("overlay")}
          badge={overlay.enabled ? "on" : "skip"}
          badgeState={overlay.enabled ? "on" : "off"}
          muted={!overlay.enabled}
        >
          <Row label="確認動画" always>
            <Check
              checked={overlay.enabled}
              disabled={busy}
              onChange={(enabled) => actions.overlay({ enabled })}
              label="proxy を生成"
            />
          </Row>
          <Row label="出力">
            <div className="checks">
              <Check
                checked={overlay.raw}
                disabled={busy}
                onChange={(raw) => actions.overlay({ raw })}
                label="推論直後"
              />
              <Check
                checked={overlay.tracked}
                disabled={busy}
                onChange={(tracked) => actions.overlay({ tracked })}
                label="追跡後"
              />
              <Check
                checked={overlay.final}
                disabled={busy}
                onChange={(final) => actions.overlay({ final })}
                label="最終マスク"
              />
              <Check
                checked={overlay.faces}
                disabled={busy || !usesFaces}
                onChange={(faces) => actions.overlay({ faces })}
                label="顔・頭部"
              />
            </div>
          </Row>
          <Row label="合成">
            <Check
              checked={overlay.finalIncludeFaces}
              disabled={busy || !usesFaces}
              onChange={(finalIncludeFaces) =>
                actions.overlay({ finalIncludeFaces })
              }
              label="最終マスクに顔を重ねる"
            />
          </Row>
          <Row label="codec">
            <Select
              value={overlay.codec}
              disabled={busy}
              onChange={(codec) => actions.overlay({ codec })}
              options={[
                { value: "mp4v", label: "mp4v — CPU" },
                { value: "h264", label: "h264 — CPU" },
                { value: "h264_nvenc", label: "h264_nvenc — GPU" },
              ]}
            />
          </Row>
          <Row label="マスク濃度">
            <Slider
              value={overlay.maskAlpha}
              min={0}
              max={1}
              step={0.01}
              disabled={busy}
              onChange={(maskAlpha) => actions.overlay({ maskAlpha })}
            />
          </Row>
          <Row label="ラベル">
            <Check
              checked={overlay.showLabels}
              disabled={busy}
              onChange={(showLabels) => actions.overlay({ showLabels })}
              label="クラス名と track ID を描画"
            />
          </Row>
          <Row label="範囲">
            <NumberInput
              value={overlay.startFrame}
              min={0}
              disabled={busy}
              onChange={(value) => actions.overlay({ startFrame: value ?? 0 })}
            />
            <NumberInput
              value={overlay.endFrame}
              min={0}
              placeholder="終端"
              disabled={busy}
              onChange={(endFrame) => actions.overlay({ endFrame })}
            />
          </Row>
          <Row label="進捗出力">
            <NumberInput
              value={overlay.progressEvery}
              min={1}
              unit="f"
              disabled={busy}
              onChange={(value) =>
                actions.overlay({ progressEvery: value ?? 1 })
              }
            />
          </Row>
        </Section>

        <Section
          name="Runtime"
          open={open.runtime}
          onToggle={() => onToggle("runtime")}
          badge={settings.backendMode === "wsl" ? "wsl2" : "native"}
        >
          <Row label="実行方式">
            <Segment
              value={settings.backendMode}
              disabled={busy}
              onChange={(backendMode) => actions.settings({ backendMode })}
              options={[
                { value: "native", label: "Native" },
                { value: "wsl", label: "WSL2" },
              ]}
            />
          </Row>
          <Row label="backend root" stack>
            <PathInput
              value={settings.backendRoot}
              placeholder="/home/user/inference_backend2"
              disabled={busy}
              browseDisabled={settings.backendMode === "wsl"}
              onChange={(backendRoot) => actions.settings({ backendRoot })}
              onBrowse={actions.pickBackendRoot}
            />
          </Row>
          <Row label="runtime python" stack>
            <PathInput
              value={settings.runtimePython}
              placeholder="/path/to/python3.10"
              disabled={busy}
              browseDisabled={settings.backendMode === "wsl"}
              onChange={(runtimePython) => actions.settings({ runtimePython })}
              onBrowse={actions.pickPython}
            />
          </Row>
          {settings.backendMode === "wsl" && (
            <Row label="distro" hint="wsl.exe -l -v で確認できます">
              <TextInput
                value={settings.wslDistro}
                placeholder="Ubuntu-24.04"
                disabled={busy}
                mono
                onChange={(wslDistro) => actions.settings({ wslDistro })}
              />
            </Row>
          )}
        </Section>
      </div>
    </Panel>
  );
}
