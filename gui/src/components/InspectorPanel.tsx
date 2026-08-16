import type {
  AppSettings,
  ClassPostprocessRule,
  InferenceMode,
  OverlayExecutionMode,
  OverlayPreset,
  PipelineDraft,
  SettingsView,
} from "../../shared/types";
import { PRODUCTION_POSTPROCESS } from "../../shared/production-contract";
import {
  FACE_MODELS,
  faceModelSpec,
  modelSpec,
  SEGMENTATION_MODELS,
} from "../lib/models";
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
  SubHead,
  TextArea,
  TextInput,
} from "./ui";

type Draft = PipelineDraft;

const CUT_METHODS = [
  { value: "high_precision", label: "高精度（推奨・FFmpeg）" },
  { value: "frame_diff", label: "フレーム差分（OpenCV）" },
];

const OVERLAY_PRESETS: ReadonlyArray<{
  value: OverlayPreset;
  label: string;
}> = [
  { value: "genital-detailed", label: "性器・詳細" },
  { value: "genital-simple", label: "性器・簡易" },
  { value: "face-detailed", label: "顔・詳細" },
  { value: "face-simple", label: "顔・簡易" },
  { value: "combined-detailed", label: "両方・詳細" },
  { value: "combined-simple", label: "両方・簡易" },
];

const SIMPLE_POSTPROCESS_CLASSES = ["男性器", "女性器", "結合部分"] as const;

function lines(value: string[]): string {
  return value.join("\n");
}

function parseLines(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

export interface InspectorActions {
  inference: (values: Partial<Draft["inference"]>) => void;
  postprocess: (values: Partial<Draft["postprocess"]>) => void;
  overlay: (values: Partial<Draft["overlay"]>) => void;
  execution: (values: Partial<Draft["execution"]>) => void;
  settings: (values: Partial<AppSettings>) => void;
  pickSqlite: (target: "inference" | "tracked" | "final") => void;
  changeMode: (mode: InferenceMode) => void;
  changeModel: (model: Draft["inference"]["segmentationModel"]) => void;
  changeFaceModel: (model: Draft["inference"]["faceModel"]) => void;
  changeOverlayExecution: (mode: OverlayExecutionMode) => void;
  pickBackendRoot: () => void;
  pickPython: () => void;
}

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
}: {
  draft: Draft;
  settings: AppSettings;
  platform: NodeJS.Platform;
  busy: boolean;
  open: Record<string, boolean>;
  viewMode: SettingsView;
  onViewModeChange: (mode: SettingsView) => void;
  onToggle: (key: string) => void;
  actions: InspectorActions;
}) {
  const { inference, postprocess, overlay } = draft;
  const advanced = viewMode === "advanced";
  const faceOnly = inference.mode === "face";
  const usesFaces = inference.mode !== "segmentation";
  const usesSegmentation = inference.mode !== "face";
  const usesRichFaces = usesFaces && inference.faceModel === "face_dino_v2";
  const facePostprocessActive =
    usesRichFaces && postprocess.faceMaskTarget !== "none";
  const legacyOverlaySelected =
    overlay.raw || overlay.tracked || overlay.final || overlay.faces;
  const hasGenitalPreset = overlay.presets.some((preset) =>
    preset.startsWith("genital-") || preset.startsWith("combined-"),
  );
  const hasFacePreset = overlay.presets.some((preset) =>
    preset.startsWith("face-") || preset.startsWith("combined-"),
  );
  const showsFaces =
    usesFaces &&
    (hasFacePreset ||
      overlay.faces ||
      (overlay.final && overlay.finalIncludeFaces));
  const mayUseEllipse =
    postprocess.classPostprocessPolicySource === "file" ||
    postprocess.shapeMode === "ellipse" ||
    (postprocess.classPostprocessPolicySource === "editor" &&
      postprocess.classPostprocessRules.some(
        (rule) => rule.shapeMode === "ellipse",
      ));
  const parallelCompatible =
    inference.mode === "segmentation-face" &&
    inference.segmentationModel === "dinov3_codino_mh0" &&
    inference.faceModel === "face_dino_v2";
  const cutMethods = CUT_METHODS.some(
    (method) => method.value === postprocess.cutMethod,
  )
    ? CUT_METHODS
    : [
        ...CUT_METHODS,
        {
          value: postprocess.cutMethod,
          label: `${postprocess.cutMethod} — カスタム`,
        },
      ];
  const spec = modelSpec(inference.segmentationModel);
  const faceSpec = faceModelSpec(inference.faceModel);

  const togglePreset = (preset: OverlayPreset, checked: boolean) => {
    if (
      !checked &&
      overlay.presets.length === 1 &&
      !legacyOverlaySelected
    ) {
      return;
    }
    actions.overlay({
      presets: checked
        ? [...overlay.presets.filter((item) => item !== preset), preset]
        : overlay.presets.filter((item) => item !== preset),
    });
  };
  const updateClassRule = (
    index: number,
    values: Partial<ClassPostprocessRule>,
  ) => {
    actions.postprocess({
      classPostprocessRules: postprocess.classPostprocessRules.map(
        (rule, ruleIndex) =>
          ruleIndex === index ? { ...rule, ...values } : rule,
      ),
    });
  };
  const simpleClassRules = SIMPLE_POSTPROCESS_CLASSES.map((className) => {
    if (postprocess.classPostprocessPolicySource === "editor") {
      const configured = postprocess.classPostprocessRules.find(
        (rule) => rule.className === className,
      );
      if (configured) {
        return configured;
      }
    }
    return {
      className,
      shapeMode: postprocess.shapeMode,
      keyframeInterval: postprocess.keyframeInterval ?? 2,
      maxGap:
        postprocess.shapeMode === "polygon"
          ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
          : (postprocess.maxGap ?? 0),
    };
  });
  const updateSimpleClassRule = (
    className: string,
    values: Partial<ClassPostprocessRule>,
  ) => {
    const rules = simpleClassRules.map((rule) =>
      rule.className === className ? { ...rule, ...values } : { ...rule },
    );
    actions.postprocess({
      classPostprocessPolicySource: "editor",
      pipelineConfig: "",
      classPostprocessPolicyJson: "",
      classPostprocessRules: rules,
    });
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

        <Section
          name="後処理"
          open={open.postprocess}
          onToggle={() => onToggle("postprocess")}
          badge={postprocess.enabled ? "性器 on" : "性器 off"}
          badgeState={postprocess.enabled ? "on" : "off"}
        >
          <Row label="性器後処理" always>
            <Check
              checked={postprocess.enabled}
              disabled={busy || faceOnly}
              onChange={(enabled) =>
                actions.postprocess({
                  enabled,
                  cutMethod:
                    !enabled && postprocess.cutDetect
                      ? "high_precision"
                      : postprocess.cutMethod,
                  precomputeCutsDuringInference:
                    !enabled &&
                    postprocess.cutDetect &&
                    inference.enabled
                      ? true
                      : postprocess.precomputeCutsDuringInference,
                })
              }
              label={faceOnly ? "顔のみでは不要" : "追跡・整形を実行"}
            />
          </Row>
          {!faceOnly && !postprocess.enabled && (
            <>
              <Row label="追跡後SQLite" stack>
                <PathInput
                  value={postprocess.trackedSqlite}
                  placeholder="tracked.sqlite"
                  disabled={busy}
                  onChange={(trackedSqlite) =>
                    actions.postprocess({ trackedSqlite })
                  }
                  onBrowse={() => actions.pickSqlite("tracked")}
                />
              </Row>
              <Row label="旧最終SQLite" stack hint="必要な場合だけ指定">
                <PathInput
                  value={postprocess.finalSqlite}
                  placeholder="predictions.sqlite"
                  disabled={busy}
                  onChange={(finalSqlite) =>
                    actions.postprocess({ finalSqlite })
                  }
                  onBrowse={() => actions.pickSqlite("final")}
                />
              </Row>
            </>
          )}
          {!faceOnly && (
            <>
              {!advanced && (
                <>
                  <SubHead>クラス別形状・キーフレーム</SubHead>
                  <Row label="クラス別設定" stack>
                    <div className="simple-policy-editor">
                      <div className="simple-policy-editor__head">
                        <span>クラス</span>
                        <span>形状</span>
                        <span>KF間隔</span>
                      </div>
                      {simpleClassRules.map((rule) => (
                        <div
                          className="simple-policy-editor__rule"
                          key={rule.className}
                        >
                          <span className="simple-policy-editor__name">
                            {rule.className}
                          </span>
                          <Segment
                            value={rule.shapeMode}
                            disabled={busy || !postprocess.enabled}
                            onChange={(shapeMode) =>
                              updateSimpleClassRule(rule.className, {
                                shapeMode,
                                maxGap:
                                  shapeMode === "polygon"
                                    ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                    : rule.maxGap,
                              })
                            }
                            options={[
                              { value: "polygon", label: "ポリゴン" },
                              { value: "ellipse", label: "楕円" },
                            ]}
                          />
                          <NumberInput
                            value={rule.keyframeInterval}
                            min={1}
                            unit="f"
                            disabled={busy || !postprocess.enabled}
                            onChange={(value) =>
                              updateSimpleClassRule(rule.className, {
                                keyframeInterval: value ?? 1,
                              })
                            }
                          />
                        </div>
                      ))}
                    </div>
                  </Row>
                </>
              )}
              {advanced &&
                postprocess.classPostprocessPolicySource !== "editor" && (
                <Row
                  label="既定形状"
                  hint={
                    postprocess.classPostprocessPolicySource === "file"
                      ? "JSONで未指定のクラスに適用"
                      : undefined
                  }
                >
                  <Segment
                    value={postprocess.shapeMode}
                    disabled={busy}
                    onChange={(shapeMode) =>
                      actions.postprocess({
                        shapeMode,
                        maxGap:
                          shapeMode === "polygon"
                            ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                            : postprocess.maxGap,
                      })
                    }
                    options={[
                      { value: "polygon", label: "ポリゴン" },
                      { value: "ellipse", label: "楕円" },
                    ]}
                  />
                </Row>
              )}
              <Row label="検出スコア下限">
                <NumberInput
                  value={postprocess.scoreMin}
                  min={0}
                  max={1}
                  step={0.01}
                  placeholder="設定既定値"
                  disabled={busy}
                  onChange={(scoreMin) =>
                    actions.postprocess({ scoreMin })
                  }
                />
              </Row>
              {advanced &&
                postprocess.classPostprocessPolicySource !== "editor" && (
                <Row
                  label="既定KF間隔"
                  hint={
                    postprocess.classPostprocessPolicySource === "file"
                      ? "JSONで未指定のクラスに適用"
                      : undefined
                  }
                >
                  <NumberInput
                    value={postprocess.keyframeInterval}
                    min={1}
                    unit="f"
                    placeholder="設定既定値"
                    disabled={busy}
                    onChange={(keyframeInterval) =>
                      actions.postprocess({ keyframeInterval })
                    }
                  />
                </Row>
              )}
            </>
          )}
          {advanced && (
            <Row
              label="カット検出"
              hint={
                !postprocess.enabled && !inference.enabled
                  ? "推論・後処理の両方を再利用する場合は実行不可"
                  : undefined
              }
            >
              <Check
                checked={postprocess.cutDetect}
                disabled={
                  busy || (!postprocess.enabled && !inference.enabled)
                }
                onChange={(cutDetect) =>
                  actions.postprocess({
                    cutDetect,
                    cutMethod:
                      !postprocess.enabled && cutDetect
                        ? "high_precision"
                        : postprocess.cutMethod,
                    precomputeCutsDuringInference:
                      cutDetect &&
                      inference.enabled &&
                      (!postprocess.enabled ||
                        postprocess.precomputeCutsDuringInference),
                  })
                }
                label="カット位置を保存し、trackを分割"
              />
            </Row>
          )}
          {usesFaces && inference.faceModel === "face_dino_v2" && (
            <>
              <Row
                label="保存する顔マスク"
                title="最終result.sqliteへ保存するプライバシーマスクです。"
              >
                <Segment
                  value={postprocess.faceMaskTarget}
                  disabled={busy}
                  onChange={(faceMaskTarget) =>
                    actions.postprocess({ faceMaskTarget })
                  }
                  options={[
                    { value: "none", label: "なし" },
                    { value: "face", label: "顔全体" },
                    { value: "eyes", label: "目元" },
                  ]}
                />
              </Row>
              {postprocess.faceMaskTarget === "eyes" && (
                <Row label="目元形状">
                  <Segment
                    value={postprocess.eyeMaskShape}
                  disabled={busy}
                  onChange={(eyeMaskShape) =>
                    actions.postprocess({ eyeMaskShape })
                  }
                    options={[
                      { value: "ellipse", label: "楕円" },
                      { value: "rectangle", label: "長方形" },
                    ]}
                  />
                </Row>
              )}
              <Row
                label="顔検出下限"
                hint="Face領域を後処理マスクへ採用する最低スコア"
              >
                <Slider
                  value={postprocess.faceDetectionScoreThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={busy || !facePostprocessActive}
                  onChange={(faceDetectionScoreThreshold) =>
                    actions.postprocess({ faceDetectionScoreThreshold })
                  }
                />
              </Row>
              <Row
                label="頭部検出下限"
                hint="Head boxを追跡へ採用する最低スコア"
              >
                <Slider
                  value={postprocess.headDetectionScoreThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={busy || !facePostprocessActive}
                  onChange={(headDetectionScoreThreshold) =>
                    actions.postprocess({ headDetectionScoreThreshold })
                  }
                />
              </Row>
            </>
          )}

          {advanced && (
            <>
              {!faceOnly && (
                <>
                  <SubHead>性器後処理 — 構成</SubHead>
                  <Row
                    label="パイプラインJSON"
                    stack
                    title="後処理stage全体を定義する上級者向け設定。クラス別形状JSONとは併用できません。"
                  >
                    <TextInput
                      value={postprocess.pipelineConfig}
                      disabled={busy || !facePostprocessActive}
                      mono
                      placeholder="空欄: 標準パイプライン"
                      onChange={(pipelineConfig) =>
                        actions.postprocess({
                          pipelineConfig,
                          classPostprocessPolicySource: pipelineConfig
                            ? "global"
                            : postprocess.classPostprocessPolicySource,
                          classPostprocessPolicyJson: pipelineConfig
                            ? ""
                            : postprocess.classPostprocessPolicyJson,
                        })
                      }
                    />
                  </Row>
                  <Row
                    label="クラス別スコアJSON"
                    stack
                    title="クラスごとの検出スコア下限を定義します。"
                  >
                    <TextInput
                      value={postprocess.classPolicyJson}
                      disabled={busy || !facePostprocessActive}
                      mono
                      placeholder="空欄: 共通の検出スコア下限"
                      onChange={(classPolicyJson) =>
                        actions.postprocess({ classPolicyJson })
                      }
                    />
                  </Row>
                  <Row
                    label="形状・KF・補完"
                    title="形状、キーフレーム間隔、補完上限の指定方法です。"
                  >
                    <Select
                      value={postprocess.classPostprocessPolicySource}
                      disabled={busy || !facePostprocessActive}
                      onChange={(classPostprocessPolicySource) =>
                        actions.postprocess({
                          classPostprocessPolicySource,
                          pipelineConfig:
                            classPostprocessPolicySource === "global"
                              ? postprocess.pipelineConfig
                              : "",
                          keyframeInterval:
                            classPostprocessPolicySource === "editor"
                              ? (postprocess.keyframeInterval ?? 3)
                              : postprocess.keyframeInterval,
                          maxGap:
                            classPostprocessPolicySource === "editor"
                              ? postprocess.shapeMode === "polygon"
                                ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                : (postprocess.maxGap ?? 0)
                              : postprocess.maxGap,
                        })
                      }
                      options={[
                        { value: "global", label: "共通値" },
                        { value: "editor", label: "クラス別GUI" },
                        { value: "file", label: "クラス別JSON" },
                      ]}
                    />
                  </Row>
                  {postprocess.classPostprocessPolicySource === "file" && (
                    <Row
                      label="形状設定JSON"
                      stack
                      title="クラス別のshape_mode・キーフレーム間隔・補完上限を持つJSONです。"
                    >
                      <TextInput
                        value={postprocess.classPostprocessPolicyJson}
                        disabled={busy}
                        mono
                        placeholder="class_postprocess_policy.json"
                        onChange={(classPostprocessPolicyJson) =>
                          actions.postprocess({
                            classPostprocessPolicyJson,
                            pipelineConfig: "",
                          })
                        }
                      />
                    </Row>
                  )}
                  {postprocess.classPostprocessPolicySource === "editor" && (
                    <Row
                      label="クラス別ルール"
                      stack
                      title="未指定クラスは先頭の「その他」行を使います。"
                    >
                      <div className="policy-editor">
                        <div className="policy-editor__head">
                          <span>確定クラス名</span>
                          <span>形状</span>
                          <span>KF間隔</span>
                          <span>補完上限</span>
                          <span />
                        </div>
                        <div className="policy-editor__rule policy-editor__rule--default">
                          <TextInput
                            value="その他（未指定）"
                            disabled
                            onChange={() => undefined}
                          />
                          <Select
                            value={postprocess.shapeMode}
                            disabled={busy}
                            onChange={(shapeMode) =>
                              actions.postprocess({
                                shapeMode,
                                maxGap:
                                  shapeMode === "polygon"
                                    ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                    : postprocess.maxGap,
                              })
                            }
                            options={[
                              { value: "polygon", label: "ポリゴン" },
                              { value: "ellipse", label: "楕円" },
                            ]}
                          />
                          <NumberInput
                            value={postprocess.keyframeInterval}
                            min={1}
                            disabled={busy}
                            onChange={(keyframeInterval) =>
                              actions.postprocess({
                                keyframeInterval:
                                  keyframeInterval ?? 1,
                              })
                            }
                          />
                          <NumberInput
                            value={
                              postprocess.shapeMode === "polygon"
                                ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                : postprocess.maxGap
                            }
                            min={0}
                            disabled={busy || postprocess.shapeMode === "polygon"}
                            onChange={(maxGap) =>
                              actions.postprocess({
                                maxGap: maxGap ?? 0,
                              })
                            }
                          />
                          <span />
                        </div>
                        {postprocess.classPostprocessRules.map(
                          (rule, index) => (
                            <div
                              className="policy-editor__rule"
                              key={index}
                            >
                              <TextInput
                                value={rule.className}
                                disabled={busy}
                                onChange={(className) =>
                                  updateClassRule(index, { className })
                                }
                              />
                              <Select
                                value={rule.shapeMode}
                                disabled={busy}
                                onChange={(shapeMode) =>
                                  updateClassRule(index, {
                                    shapeMode,
                                    maxGap:
                                      shapeMode === "polygon"
                                        ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                        : rule.maxGap,
                                  })
                                }
                                options={[
                                  { value: "polygon", label: "ポリゴン" },
                                  { value: "ellipse", label: "楕円" },
                                ]}
                              />
                              <NumberInput
                                value={rule.keyframeInterval}
                                min={1}
                                disabled={busy}
                                onChange={(value) =>
                                  updateClassRule(index, {
                                    keyframeInterval: value ?? 1,
                                  })
                                }
                              />
                              <NumberInput
                                value={
                                  rule.shapeMode === "polygon"
                                    ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                    : rule.maxGap
                                }
                                min={0}
                                disabled={busy || rule.shapeMode === "polygon"}
                                onChange={(value) =>
                                  updateClassRule(index, {
                                    maxGap: value ?? 0,
                                  })
                                }
                              />
                              <button
                                type="button"
                                className="btn btn--sm btn--quiet"
                                disabled={busy}
                                title={`${rule.className || "クラス"}を削除`}
                                onClick={() =>
                                  actions.postprocess({
                                    classPostprocessRules:
                                      postprocess.classPostprocessRules.filter(
                                        (_, ruleIndex) =>
                                          ruleIndex !== index,
                                      ),
                                  })
                                }
                              >
                                −
                              </button>
                            </div>
                          ),
                        )}
                        <button
                          type="button"
                          className="btn btn--sm btn--quiet policy-editor__add"
                          disabled={busy}
                          onClick={() =>
                            actions.postprocess({
                              classPostprocessRules: [
                                ...postprocess.classPostprocessRules,
                                {
                                  className: `class_${
                                    postprocess.classPostprocessRules.length + 1
                                  }`,
                                  shapeMode: postprocess.shapeMode,
                                  keyframeInterval:
                                    postprocess.keyframeInterval ?? 3,
                                  maxGap:
                                    postprocess.shapeMode === "polygon"
                                      ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                                      : (postprocess.maxGap ?? 0),
                                },
                              ],
                            })
                          }
                        >
                          ＋ クラスを追加
                        </button>
                      </div>
                    </Row>
                  )}
                </>
              )}
              <SubHead>カット検出</SubHead>
              <Row
                label="検出方式"
                hint={
                  postprocess.enabled
                    ? postprocess.cutMethod
                    : "後処理なしではhigh_precisionのみ"
                }
              >
                <Select
                  value={postprocess.cutMethod}
                  disabled={
                    busy || !postprocess.cutDetect || !postprocess.enabled
                  }
                  onChange={(cutMethod) =>
                    actions.postprocess({
                      cutMethod,
                      precomputeCutsDuringInference:
                        cutMethod === "high_precision"
                          ? postprocess.precomputeCutsDuringInference
                          : false,
                    })
                  }
                  options={cutMethods}
                />
              </Row>
              <Row
                label="推論と同時検出"
                title="別のFFmpeg縮小decodeをCPUで動かし、GPU推論時間へ重ねます。"
              >
                <Check
                  checked={postprocess.precomputeCutsDuringInference}
                  disabled={
                    busy ||
                    !inference.enabled ||
                    !postprocess.cutDetect ||
                    postprocess.cutMethod !== "high_precision" ||
                    !postprocess.enabled
                  }
                  onChange={(precomputeCutsDuringInference) =>
                    actions.postprocess({
                      precomputeCutsDuringInference,
                    })
                  }
                  label={
                    !postprocess.enabled
                      ? "後処理なしでは同時検出が必須"
                      : postprocess.cutMethod === "high_precision"
                      ? "CPU検出をGPU推論と並行"
                      : "high_precisionのみ対応"
                  }
                />
              </Row>
              {!faceOnly && (
                <>
                  <SubHead>性器後処理 — 追跡・補完</SubHead>
                  <Row
                    label="短命track上限"
                    hint="指定フレーム以下のtrackを除去"
                  >
                    <NumberInput
                      value={postprocess.removeShortTracksMaxFrames}
                      min={0}
                      unit="f"
                      placeholder="既定値"
                      disabled={busy || !facePostprocessActive}
                      onChange={(removeShortTracksMaxFrames) =>
                        actions.postprocess({
                          removeShortTracksMaxFrames,
                        })
                      }
                    />
                  </Row>
                  {postprocess.classPostprocessPolicySource !== "editor" && (
                    <Row
                      label="既定補完上限"
                      hint={
                        postprocess.shapeMode === "polygon"
                          ? "Productionポリゴンは15フレーム固定"
                          : undefined
                      }
                    >
                      <NumberInput
                        value={
                          postprocess.shapeMode === "polygon"
                            ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
                            : postprocess.maxGap
                        }
                        min={0}
                        unit="f"
                        placeholder="既定値"
                        disabled={busy || postprocess.shapeMode === "polygon"}
                        onChange={(maxGap) =>
                          actions.postprocess({ maxGap })
                        }
                      />
                    </Row>
                  )}
                  <SubHead>楕円近似（K2）</SubHead>
                  <Row
                    label="モデルroot"
                    stack
                    off={!mayUseEllipse}
                    title="楕円を使用する場合だけ有効です。"
                  >
                    <TextInput
                      value={postprocess.modelRoot}
                      disabled={busy || !mayUseEllipse}
                      mono
                      placeholder="空欄: 自動検出"
                      onChange={(modelRoot) =>
                        actions.postprocess({ modelRoot })
                      }
                    />
                  </Row>
                  <Row label="K2 run directory" stack off={!mayUseEllipse}>
                    <TextInput
                      value={postprocess.k2RunDir}
                      disabled={busy || !mayUseEllipse}
                      mono
                      placeholder="空欄: model root/k2_v5"
                      onChange={(k2RunDir) =>
                        actions.postprocess({ k2RunDir })
                      }
                    />
                  </Row>
                  <Row label="GPUバッチ数" off={!mayUseEllipse}>
                    <NumberInput
                      value={postprocess.k2BatchSize}
                      min={1}
                      placeholder="パイプライン既定"
                      disabled={busy || !mayUseEllipse}
                      onChange={(k2BatchSize) =>
                        actions.postprocess({ k2BatchSize })
                      }
                    />
                  </Row>
                  <Row label="CPU前処理worker" off={!mayUseEllipse}>
                    <NumberInput
                      value={postprocess.k2PrepWorkers}
                      min={0}
                      placeholder="既定値"
                      disabled={busy || !mayUseEllipse}
                      onChange={(k2PrepWorkers) =>
                        actions.postprocess({ k2PrepWorkers })
                      }
                    />
                  </Row>
                  <Row label="計算精度" off={!mayUseEllipse}>
                    <Select
                      value={postprocess.k2Precision ?? ""}
                      disabled={busy || !mayUseEllipse}
                      onChange={(value) =>
                        actions.postprocess({
                          k2Precision: value === "" ? null : value,
                        })
                      }
                      options={[
                        { value: "", label: "パイプライン既定" },
                        { value: "fp32", label: "FP32" },
                        { value: "fp16", label: "FP16" },
                      ]}
                    />
                  </Row>
                  <Row
                    label="計算範囲"
                    off={!mayUseEllipse}
                    title="states_onlyは未使用のソフトマスク生成を省略します。"
                  >
                    <Select
                      value={postprocess.k2ForwardMode ?? ""}
                      disabled={busy || !mayUseEllipse}
                      onChange={(value) =>
                        actions.postprocess({
                          k2ForwardMode: value === "" ? null : value,
                        })
                      }
                      options={[
                        { value: "", label: "パイプライン既定" },
                        { value: "states_only", label: "必要値のみ（推奨）" },
                        { value: "full", label: "全出力（診断用）" },
                      ]}
                    />
                  </Row>
                  <Row
                    label="内部時間計測"
                    off={!mayUseEllipse}
                    title="正確な計測のためGPU同期が入り、通常処理は遅くなります。"
                  >
                    <Select
                      value={
                        postprocess.k2ProfileStages === null
                          ? ""
                          : postprocess.k2ProfileStages
                            ? "on"
                            : "off"
                      }
                      disabled={busy || !mayUseEllipse}
                      onChange={(value) =>
                        actions.postprocess({
                          k2ProfileStages:
                            value === "" ? null : value === "on",
                        })
                      }
                      options={[
                        { value: "", label: "パイプライン既定" },
                        { value: "on", label: "計測する" },
                        { value: "off", label: "計測しない（推奨）" },
                      ]}
                    />
                  </Row>
                  <Row label="cuDNN autotune" off={!mayUseEllipse}>
                    <Select
                      value={postprocess.k2CudnnBenchmark ?? ""}
                      disabled={busy || !mayUseEllipse}
                      onChange={(value) =>
                        actions.postprocess({
                          k2CudnnBenchmark:
                            value === "" ? null : value,
                        })
                      }
                      options={[
                        { value: "", label: "パイプライン既定" },
                        { value: "on", label: "有効" },
                        { value: "off", label: "無効" },
                      ]}
                    />
                  </Row>
                  <Row label="TF32" off={!mayUseEllipse}>
                    <Select
                      value={postprocess.k2Tf32 ?? ""}
                      disabled={busy || !mayUseEllipse}
                      onChange={(value) =>
                        actions.postprocess({
                          k2Tf32: value === "" ? null : value,
                        })
                      }
                      options={[
                        { value: "", label: "パイプライン既定" },
                        { value: "default", label: "PyTorch既定" },
                        { value: "on", label: "有効" },
                        { value: "off", label: "無効（再現性優先）" },
                      ]}
                    />
                  </Row>
                  <Row label="K2デバイス" off={!mayUseEllipse}>
                    <TextInput
                      value={postprocess.device}
                      disabled={busy || !mayUseEllipse}
                      mono
                      onChange={(device) =>
                        actions.postprocess({ device })
                      }
                    />
                  </Row>
                  <SubHead>互換出力</SubHead>
                  <Row
                    label="旧形式SQLite"
                    title="最新result.sqliteに加え、旧Dinov3_postprocess互換ファイルを追加します。"
                  >
                    <Check
                      checked={postprocess.exportLegacySqlite}
                      disabled={busy || !postprocess.enabled}
                      onChange={(exportLegacySqlite) =>
                        actions.postprocess({ exportLegacySqlite })
                      }
                      label="旧形式も追加"
                    />
                  </Row>
                </>
              )}
              {usesFaces && inference.faceModel === "face_dino_v2" && (
                <>
                  <SubHead>顔後処理 — 追跡・プライバシーマスク</SubHead>
                  <Row label="目キーポイント下限">
                    <Slider
                      value={postprocess.minimumEyeConfidence}
                      min={0}
                      max={1}
                      step={0.01}
                      disabled={busy || !facePostprocessActive}
                      onChange={(minimumEyeConfidence) =>
                        actions.postprocess({ minimumEyeConfidence })
                      }
                    />
                  </Row>
                  <Row
                    label="追跡保持gap"
                    hint="未検出を許容する最大フレーム数"
                  >
                    <NumberInput
                      value={postprocess.faceTrackingMaxGapFrames}
                      min={0}
                      unit="f"
                      disabled={busy || !facePostprocessActive}
                      onChange={(value) =>
                        actions.postprocess({
                          faceTrackingMaxGapFrames: value ?? 0,
                        })
                      }
                    />
                  </Row>
                  <Row label="追跡high閾値">
                    <Slider
                      value={postprocess.faceTrackingHighScoreThreshold}
                      min={postprocess.faceTrackingLowScoreThreshold}
                      max={1}
                      step={0.01}
                      disabled={busy || !facePostprocessActive}
                      onChange={(faceTrackingHighScoreThreshold) =>
                        actions.postprocess({
                          faceTrackingHighScoreThreshold,
                        })
                      }
                    />
                  </Row>
                  <Row label="追跡low閾値">
                    <Slider
                      value={postprocess.faceTrackingLowScoreThreshold}
                      min={0}
                      max={postprocess.faceTrackingHighScoreThreshold}
                      step={0.01}
                      disabled={busy || !facePostprocessActive}
                      onChange={(faceTrackingLowScoreThreshold) =>
                        actions.postprocess({
                          faceTrackingLowScoreThreshold,
                        })
                      }
                    />
                  </Row>
                  <Row
                    label="短命track上限"
                    hint="観測回数（hits）で判定"
                  >
                    <NumberInput
                      value={postprocess.faceShortTrackMaxHits}
                      min={0}
                      disabled={busy || !facePostprocessActive}
                      onChange={(value) =>
                        actions.postprocess({
                          faceShortTrackMaxHits: value ?? 0,
                        })
                      }
                    />
                  </Row>
                  <Row label="短命保持スコア">
                    <Slider
                      value={postprocess.faceShortTrackKeepScore}
                      min={0}
                      max={1}
                      step={0.01}
                      disabled={busy || !facePostprocessActive}
                      onChange={(faceShortTrackKeepScore) =>
                        actions.postprocess({ faceShortTrackKeepScore })
                      }
                    />
                  </Row>
                  <Row label="補完gap上限">
                    <NumberInput
                      value={postprocess.faceInterpolationMaxGap}
                      min={0}
                      unit="f"
                      disabled={busy || !facePostprocessActive}
                      onChange={(value) =>
                        actions.postprocess({
                          faceInterpolationMaxGap: value ?? 0,
                        })
                      }
                    />
                  </Row>
                </>
              )}
              <SubHead>専門設定</SubHead>
              <Row
                label="追加CLI引数"
                stack
                title="未型付けの将来オプション用です。上の管理済み引数は上書きできません。"
              >
                <TextArea
                  value={lines(postprocess.extraArgs)}
                  disabled={busy}
                  placeholder={"--future-option\nvalue"}
                  onChange={(value) =>
                    actions.postprocess({ extraArgs: parseLines(value) })
                  }
                />
              </Row>
            </>
          )}
        </Section>

        <Section
          name="オーバーレイ"
          open={open.overlay}
          onToggle={() => onToggle("overlay")}
          badge={overlay.enabled ? overlay.executionMode : "skip"}
          badgeState={overlay.enabled ? "on" : "off"}
        >
          <Row label="確認動画" always>
            <Check
              checked={overlay.enabled}
              disabled={busy}
              onChange={(enabled) => actions.overlay({ enabled })}
              label="オーバーレイを生成"
            />
          </Row>
          <Row label="表示プリセット" stack>
            <div className="checks">
              {OVERLAY_PRESETS.map((preset) => {
                const needsFace = preset.value.startsWith("face-") ||
                  preset.value.startsWith("combined-");
                const needsSegmentation =
                  preset.value.startsWith("genital-") ||
                  preset.value.startsWith("combined-");
                return (
                  <Check
                    key={preset.value}
                    checked={overlay.presets.includes(preset.value)}
                    disabled={
                      busy ||
                      (needsFace && !usesFaces) ||
                      (needsSegmentation && !usesSegmentation)
                    }
                    onChange={(checked) =>
                      togglePreset(preset.value, checked)
                    }
                    label={preset.label}
                  />
                );
              })}
            </div>
          </Row>
          <Row label="エンコード">
            <Segment<OverlayExecutionMode>
              value={overlay.executionMode}
              disabled={busy}
              onChange={actions.changeOverlayExecution}
              options={[
                { value: "cpu", label: "CPU" },
                { value: "nvenc", label: "NVENC" },
                { value: "fast", label: "高速" },
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

          {advanced && (
            <>
              <SubHead>旧形式の追加オーバーレイ（任意）</SubHead>
              <Row
                label="追加動画"
                stack
                title="通常の表示プリセットとは別に、旧ソフト互換の工程別MP4を追加生成します。SQLiteの内容は増えません。"
              >
                <div className="checks">
                  <Check
                    checked={overlay.raw}
                    disabled={busy || !usesSegmentation}
                    onChange={(raw) => actions.overlay({ raw })}
                    label="AI生出力（raw.mp4）"
                  />
                  <Check
                    checked={overlay.tracked}
                    disabled={busy || !usesSegmentation}
                    onChange={(tracked) => actions.overlay({ tracked })}
                    label="追跡後（tracked.mp4）"
                  />
                  <Check
                    checked={overlay.final}
                    disabled={busy}
                    onChange={(final) => actions.overlay({ final })}
                    label="最終後処理（final.mp4）"
                  />
                  <Check
                    checked={overlay.faces}
                    disabled={busy || !usesFaces}
                    onChange={(faces) => actions.overlay({ faces })}
                    label="顔box（faces.mp4）"
                  />
                </div>
              </Row>
              <Row label="finalへ顔を追加">
                <Check
                  checked={overlay.finalIncludeFaces}
                  disabled={
                    busy ||
                    !usesFaces ||
                    !usesSegmentation ||
                    !overlay.final
                  }
                  onChange={(finalIncludeFaces) =>
                    actions.overlay({ finalIncludeFaces })
                  }
                  label="互換finalへ顔boxを合成"
                />
              </Row>
              <SubHead>描画内容</SubHead>
              {hasGenitalPreset && (
                <Row
                  label="性器の描画データ"
                  title="性器・両方プリセットへ描くデータを選びます。AI生マスクは後処理前、最終マスクは追跡・補完・形状近似などの後処理後です。SQLite自体は変更しません。"
                >
                  <Segment
                    value={overlay.genitalSource}
                    disabled={busy}
                    onChange={(genitalSource) =>
                      actions.overlay({ genitalSource })
                    }
                    options={[
                      { value: "raw", label: "AI生マスク（後処理前）" },
                      { value: "final", label: "最終マスク（後処理後）" },
                    ]}
                  />
                </Row>
              )}
              {showsFaces && usesRichFaces && (
                <>
                  <Row
                    label="描画時追加マスク"
                    title="確認動画だけに顔/目元マスクを追加します。result.sqliteの保存内容は変更しません。"
                  >
                    <Segment
                      value={overlay.faceMaskTarget}
                      disabled={busy}
                      onChange={(faceMaskTarget) =>
                        actions.overlay({ faceMaskTarget })
                      }
                      options={[
                        { value: "none", label: "なし" },
                        { value: "face", label: "顔全体" },
                        { value: "eyes", label: "目元" },
                      ]}
                    />
                  </Row>
                  <Row label="追加目元マスク形状">
                    <Segment
                      value={overlay.eyeMaskShape}
                      disabled={busy || overlay.faceMaskTarget !== "eyes"}
                      onChange={(eyeMaskShape) =>
                        actions.overlay({ eyeMaskShape })
                      }
                      options={[
                        { value: "ellipse", label: "楕円" },
                        { value: "rectangle", label: "長方形" },
                      ]}
                    />
                  </Row>
                  <Row label="目キーポイント下限">
                    <Slider
                      value={overlay.minimumEyeConfidence}
                      min={0}
                      max={1}
                      step={0.01}
                      disabled={
                        busy || overlay.faceMaskTarget !== "eyes"
                      }
                      onChange={(minimumEyeConfidence) =>
                        actions.overlay({ minimumEyeConfidence })
                      }
                    />
                  </Row>
                  <Row label="顔詳細要素" stack>
                    <div className="checks">
                      <Check
                        checked={overlay.faceProbabilityMasks}
                        disabled={busy}
                        onChange={(faceProbabilityMasks) =>
                          actions.overlay({ faceProbabilityMasks })
                        }
                        label="確率マスク"
                      />
                      <Check
                        checked={overlay.faceKeypoints}
                        disabled={busy}
                        onChange={(faceKeypoints) =>
                          actions.overlay({ faceKeypoints })
                        }
                        label="キーポイント"
                      />
                      <Check
                        checked={overlay.faceEllipses}
                        disabled={busy}
                        onChange={(faceEllipses) =>
                          actions.overlay({ faceEllipses })
                        }
                        label="顔楕円"
                      />
                    </div>
                  </Row>
                </>
              )}
              <Row label="線幅">
                <NumberInput
                  value={overlay.outlineThickness}
                  min={1}
                  unit="マスク"
                  disabled={busy}
                  onChange={(value) =>
                    actions.overlay({ outlineThickness: value ?? 1 })
                  }
                />
                <NumberInput
                  value={overlay.boxThickness}
                  min={1}
                  unit="box"
                  disabled={busy}
                  onChange={(value) =>
                    actions.overlay({ boxThickness: value ?? 1 })
                  }
                />
              </Row>
              <Row label="ラベル">
                <Check
                  checked={overlay.showLabels}
                  disabled={busy}
                  onChange={(showLabels) =>
                    actions.overlay({ showLabels })
                  }
                  label="クラス・確信度・track ID"
                />
              </Row>
              <SubHead>エンコード品質・速度</SubHead>
              {overlay.executionMode === "cpu" && (
                <>
                  <Row
                    label="x264 CRF"
                    hint={
                      overlay.targetBitrateMbps === null
                        ? "小さいほど高画質"
                        : "ビットレート指定中は不使用"
                    }
                  >
                    <NumberInput
                      value={overlay.h264Crf}
                      min={0}
                      max={51}
                      disabled={busy || overlay.targetBitrateMbps !== null}
                      onChange={(value) =>
                        actions.overlay({ h264Crf: value ?? 18 })
                      }
                    />
                  </Row>
                </>
              )}
              {(overlay.executionMode === "cpu" ||
                overlay.executionMode === "fast") && (
                <Row
                  label={
                    overlay.executionMode === "fast"
                      ? "CPU preset"
                      : "x264 preset"
                  }
                >
                    <Select
                      value={overlay.h264Preset}
                      disabled={busy}
                      onChange={(h264Preset) =>
                        actions.overlay({ h264Preset })
                      }
                      options={(
                        [
                          "ultrafast",
                          "superfast",
                          "veryfast",
                          "faster",
                          "fast",
                          "medium",
                          "slow",
                          "slower",
                          "veryslow",
                        ] as const
                      ).map((value) => ({ value, label: value }))}
                    />
                </Row>
              )}
              <Row label="FFmpeg実行ファイル" stack>
                <TextInput
                  value={overlay.ffmpegBin}
                  disabled={busy}
                  mono
                  placeholder="空欄: 同梱FFmpeg"
                  onChange={(ffmpegBin) =>
                    actions.overlay({ ffmpegBin })
                  }
                />
              </Row>
              {overlay.executionMode === "nvenc" && (
                <>
                  <Row
                    label="NVENC CQ"
                    hint={
                      overlay.targetBitrateMbps === null
                        ? "小さいほど高画質"
                        : "ビットレート指定中は不使用"
                    }
                  >
                    <NumberInput
                      value={overlay.nvencCq}
                      min={0}
                      max={51}
                      disabled={busy || overlay.targetBitrateMbps !== null}
                      onChange={(value) =>
                        actions.overlay({ nvencCq: value ?? 18 })
                      }
                    />
                  </Row>
                  <Row label="NVENC preset">
                    <Select
                      value={overlay.nvencPreset}
                      disabled={busy}
                      onChange={(nvencPreset) =>
                        actions.overlay({ nvencPreset })
                      }
                      options={(
                        [
                          "p1",
                          "p2",
                          "p3",
                          "p4",
                          "p5",
                          "p6",
                          "p7",
                        ] as const
                      ).map((value) => ({ value, label: value }))}
                    />
                  </Row>
                  <Row label="NVENC GPU">
                    <NumberInput
                      value={overlay.nvencGpu}
                      min={0}
                      disabled={busy}
                      onChange={(value) =>
                        actions.overlay({ nvencGpu: value ?? 0 })
                      }
                    />
                  </Row>
                </>
              )}
              {overlay.executionMode === "fast" && (
                <>
                  <Row label="NVENC preset">
                    <Select
                      value={overlay.nvencPreset}
                      disabled={busy}
                      onChange={(nvencPreset) =>
                        actions.overlay({ nvencPreset })
                      }
                      options={(
                        ["p1", "p2", "p3", "p4", "p5", "p6", "p7"] as const
                      ).map((value) => ({ value, label: value }))}
                    />
                  </Row>
                  <Row label="NVENC GPU">
                    <NumberInput
                      value={overlay.nvencGpu}
                      min={0}
                      disabled={busy}
                      onChange={(value) =>
                        actions.overlay({ nvencGpu: value ?? 0 })
                      }
                    />
                  </Row>
                </>
              )}
              <Row
                label="目標ビットレート"
                hint={
                  overlay.executionMode === "fast"
                    ? "高速モードでは必須（空欄時8 Mbps）"
                    : "空欄ならCRF/CQ品質指定"
                }
              >
                <NumberInput
                  value={overlay.targetBitrateMbps}
                  min={0.1}
                  step={0.1}
                  unit="Mbps"
                  placeholder={
                    overlay.executionMode === "fast"
                      ? "8.0"
                      : "空欄: CRF/CQ"
                  }
                  disabled={busy}
                  onChange={(targetBitrateMbps) =>
                    actions.overlay({ targetBitrateMbps })
                  }
                />
              </Row>
              {overlay.executionMode === "fast" && (
                <>
                  <Row label="分割worker総数">
                    <NumberInput
                      value={overlay.workers}
                      min={1}
                      disabled={busy}
                      onChange={(value) =>
                        actions.overlay({ workers: value ?? 1 })
                      }
                    />
                  </Row>
                  <Row
                    label="CPU割当数"
                    hint="残りをNVENCへ割当"
                  >
                    <NumberInput
                      value={overlay.cpuWorkers}
                      min={0}
                      max={overlay.workers}
                      disabled={busy}
                      onChange={(value) =>
                        actions.overlay({ cpuWorkers: value ?? 0 })
                      }
                    />
                  </Row>
                  <Row label="音声保持">
                    <Check
                      checked={overlay.copyAudio}
                      disabled={busy}
                      onChange={(copyAudio) =>
                        actions.overlay({ copyAudio })
                      }
                      label="元音声をコピー"
                    />
                  </Row>
                  <Row label="MP4 faststart">
                    <Check
                      checked={overlay.faststart}
                      disabled={busy}
                      onChange={(faststart) =>
                        actions.overlay({ faststart })
                      }
                      label="Web再生用に最適化"
                    />
                  </Row>
                </>
              )}
              <SubHead>処理範囲・ログ</SubHead>
              <Row label="フレーム範囲" hint="開始 / 終了">
                <NumberInput
                  value={overlay.startFrame}
                  min={0}
                  disabled={busy}
                  onChange={(value) =>
                    actions.overlay({ startFrame: value ?? 0 })
                  }
                />
                <NumberInput
                  value={overlay.endFrame}
                  min={overlay.startFrame}
                  placeholder="終端"
                  disabled={busy}
                  onChange={(endFrame) =>
                    actions.overlay({ endFrame })
                  }
                />
              </Row>
              <Row
                label="進捗ログ間隔"
                hint={
                  overlay.executionMode === "fast"
                    ? "分割高速モードでは内部管理"
                    : "0で無効"
                }
              >
                <NumberInput
                  value={overlay.progressEvery}
                  min={0}
                  unit="f"
                  disabled={busy || overlay.executionMode === "fast"}
                  onChange={(value) =>
                    actions.overlay({ progressEvery: value ?? 0 })
                  }
                />
              </Row>
              <SubHead>専門設定</SubHead>
              <Row
                label="追加CLI引数"
                stack
                title="未型付けの将来オプション用です。上の管理済み引数は上書きできません。"
              >
                <TextArea
                  value={lines(overlay.extraArgs)}
                  disabled={busy}
                  placeholder={"--future-option\nvalue"}
                  onChange={(value) =>
                    actions.overlay({ extraArgs: parseLines(value) })
                  }
                />
              </Row>
            </>
          )}
        </Section>

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
      </div>
    </Panel>
  );
}
