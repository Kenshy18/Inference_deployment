import type { InferenceMode } from "../../../shared/types";
import {
  FACE_MODELS,
  faceModelSpec,
  modelSpec,
  SEGMENTATION_MODELS,
} from "../../lib/models";
import {
  Check,
  NumberInput,
  PathInput,
  Row,
  Section,
  Segment,
  Select,
  Slider,
  SubHead,
  TextArea,
  TextInput,
} from "../ui";
import { lines, parseLines } from "./shared";
import type { InspectorSectionProps } from "./types";


export function InferenceSection({
  draft,
  busy,
  open,
  advanced,
  onToggle,
  actions,
}: InspectorSectionProps) {
  const { inference, postprocess } = draft;
  const usesFaces = inference.mode !== "segmentation";
  const usesSegmentation = inference.mode !== "face";
  const parallelCompatible =
    inference.mode === "segmentation-face" &&
    inference.segmentationModel === "dinov3_codino_mh0" &&
    inference.faceModel === "face_dino_v2";
  const spec = modelSpec(inference.segmentationModel);
  const faceSpec = faceModelSpec(inference.faceModel);

  return (
        <Section
          name="推論"
          open={open.inference}
          onToggle={() => onToggle("inference")}
          badge={inference.enabled ? "on" : "reuse"}
          badgeState={inference.enabled ? "on" : "off"}
        >
          <Row label="推論" always>
            <Check
              checked={inference.enabled}
              disabled={busy}
              onChange={(enabled) => {
                actions.inference({
                  enabled,
                  parallelModels: enabled
                    ? inference.parallelModels
                    : false,
                  parallelModelStaggerSeconds: enabled
                    ? inference.parallelModelStaggerSeconds
                    : 0,
                });
                if (!enabled) {
                  actions.postprocess({
                    precomputeCutsDuringInference: false,
                  });
                }
              }}
              label="新規に実行"
            />
          </Row>
          {!inference.enabled && (
            <Row
              label="AI推論SQLite"
              stack
              hint="生成済みunified inference SQLiteをキューの全動画で再利用します"
            >
              <PathInput
                value={inference.inputSqlite}
                placeholder="inference.sqlite"
                disabled={busy}
                onChange={(inputSqlite) =>
                  actions.inference({ inputSqlite })
                }
                onBrowse={() => actions.pickSqlite("inference")}
              />
            </Row>
          )}
          <Row label="処理モード" always>
            <Segment<InferenceMode>
              value={inference.mode}
              disabled={busy}
              onChange={actions.changeMode}
              options={[
                { value: "segmentation", label: "性器" },
                { value: "segmentation-face", label: "両方" },
                { value: "face", label: "顔" },
              ]}
            />
          </Row>
          {usesSegmentation && (
            <>
              <Row label="性器モデル">
                <Select
                  value={inference.segmentationModel}
                  disabled={busy || !inference.enabled}
                  onChange={actions.changeModel}
                  options={SEGMENTATION_MODELS.map((model) => ({
                    value: model.id,
                    label: model.label,
                  }))}
                />
              </Row>
              <Row label="推論エンジン">
                <Select
                  value={inference.segmentationBackend}
                  disabled={
                    busy || !inference.enabled || spec.backends.length < 2
                  }
                  onChange={(segmentationBackend) =>
                    actions.inference({ segmentationBackend })
                  }
                  options={spec.backends.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                />
              </Row>
            </>
          )}
          {usesFaces && (
            <>
              <Row label="顔モデル">
                <Select
                  value={inference.faceModel}
                  disabled={busy || !inference.enabled}
                  onChange={actions.changeFaceModel}
                  options={FACE_MODELS.map((model) => ({
                    value: model.id,
                    label: model.label,
                  }))}
                />
              </Row>
              <Row
                label="顔推論エンジン"
                title="選択中の顔モデルを実行するバックエンドです。現行モデルは新顔=TensorRT、旧顔=PyTorchに固定されています。"
              >
                <Select
                  value={inference.faceBackend}
                  disabled={
                    busy ||
                    !inference.enabled ||
                    faceSpec.backends.length < 2
                  }
                  onChange={(faceBackend) =>
                    actions.inference({ faceBackend })
                  }
                  options={faceSpec.backends.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                />
              </Row>
            </>
          )}

          {advanced && (
            <>
              <SubHead>モデル入力・実行範囲</SubHead>
              {usesFaces && (
                <>
                  <Row
                    label="顔検出の保存対象"
                    stack
                    title="顔モデルの出力からSQLiteへ保存する対象です。Headは頭部box、Faceは顔領域です。少なくとも1つを選択します。"
                  >
                    <div className="checks">
                      {[
                        {
                          value: "Head",
                          label: "Head（頭部box）",
                        },
                        {
                          value: "Face",
                          label: "Face（顔領域）",
                        },
                      ].map((option) => {
                        const checked = inference.faceClasses.includes(
                          option.value,
                        );
                        return (
                          <Check
                            key={option.value}
                            checked={checked}
                            disabled={busy || !inference.enabled}
                            onChange={(nextChecked) => {
                              const faceClasses = nextChecked
                                ? [
                                    ...inference.faceClasses,
                                    option.value,
                                  ]
                                : inference.faceClasses.filter(
                                    (value) => value !== option.value,
                                  );
                              if (faceClasses.length > 0) {
                                actions.inference({ faceClasses });
                              }
                            }}
                            label={option.label}
                          />
                        );
                      })}
                    </div>
                  </Row>
                  {inference.faceModel === "face_dino_v2" && (
                    <Row
                      label="顔TRT bundle"
                      stack
                      title="新顔モデルのTensorRT bundleだけを上書きします。"
                    >
                      <TextInput
                        value={inference.faceTrtBundle}
                        disabled={busy || !inference.enabled}
                        mono
                        placeholder="空欄: 自動選択"
                        onChange={(faceTrtBundle) =>
                          actions.inference({ faceTrtBundle })
                        }
                      />
                    </Row>
                  )}
                </>
              )}
              <Row label="推論デバイス">
                <TextInput
                  value={inference.device}
                  disabled={busy || !inference.enabled}
                  mono
                  onChange={(device) => actions.inference({ device })}
                />
              </Row>
              <Row label="処理上限">
                <NumberInput
                  value={inference.maxFrames}
                  min={1}
                  placeholder="空欄: 全フレーム"
                  disabled={busy || !inference.enabled}
                  onChange={(maxFrames) => actions.inference({ maxFrames })}
                />
              </Row>
              <Row label="速度計測除外">
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
                    unit="回"
                    disabled={busy || !inference.enabled}
                    onChange={(value) =>
                      actions.inference({
                        faceWarmupIterations: value ?? 0,
                      })
                    }
                  />
                </Row>
              )}
              <SubHead>性能</SubHead>
              <Row
                label="モデル同時推論"
                hint={
                  parallelCompatible
                    ? "v3-lite + Face V2限定"
                    : "現在の組合せでは不可"
                }
              >
                <Check
                  checked={inference.parallelModels}
                  disabled={
                    busy || !inference.enabled || !parallelCompatible
                  }
                  onChange={(parallelModels) =>
                    actions.inference({
                      parallelModels,
                      parallelModelStaggerSeconds: parallelModels
                        ? inference.parallelModelStaggerSeconds
                        : 0,
                    })
                  }
                  label="性器・顔モデルを同時実行"
                />
              </Row>
              {inference.parallelModels && (
                <Row
                  label="顔→性器の開始差"
                  hint="0秒が実測上の推奨値"
                >
                  <NumberInput
                    value={inference.parallelModelStaggerSeconds}
                    min={0}
                    step={0.1}
                    unit="秒"
                    disabled={busy}
                    onChange={(value) =>
                      actions.inference({
                        parallelModelStaggerSeconds: value ?? 0,
                      })
                    }
                  />
                </Row>
              )}
              <Row
                label="SQLite書き込み"
                hint="高速化する代わりに異常終了時の耐性が低下"
              >
                <Check
                  checked={inference.fastSqlite}
                  disabled={busy || !inference.enabled}
                  onChange={(fastSqlite) =>
                    actions.inference({ fastSqlite })
                  }
                  label="速度優先モード"
                />
              </Row>
              <SubHead>専門設定</SubHead>
              <Row
                label="追加引数"
                stack
                title="将来の未型付け引数を1トークン1行で指定します。主要引数の上書きは禁止されます。"
              >
                <TextArea
                  value={lines(inference.extraArgs)}
                  disabled={busy}
                  placeholder={"--future-option\nvalue"}
                  onChange={(value) =>
                    actions.inference({ extraArgs: parseLines(value) })
                  }
                />
              </Row>
            </>
          )}
        </Section>
  );
}
