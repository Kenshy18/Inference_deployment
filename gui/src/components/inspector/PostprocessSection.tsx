import type { ClassPostprocessRule } from "../../../shared/types";
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
import {
  CUT_METHODS,
  SIMPLE_POSTPROCESS_CLASSES,
  lines,
  parseLines,
} from "./shared";
import type { InspectorSectionProps } from "./types";


export function PostprocessSection({
  draft,
  busy,
  open,
  advanced,
  onToggle,
  actions,
}: InspectorSectionProps) {
  const { inference, postprocess } = draft;
  const faceOnly = inference.mode === "face";
  const usesFaces = inference.mode !== "segmentation";
  const usesRichFaces = usesFaces && inference.faceModel === "face_dino_v2";
  const facePostprocessActive =
    usesRichFaces && postprocess.faceMaskTarget !== "none";
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
      keyframeInterval: postprocess.keyframeInterval ?? 2,
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
                  <SubHead>クラス別キーフレーム</SubHead>
                  <Row label="クラス別設定" stack>
                    <div className="simple-policy-editor">
                      <div className="simple-policy-editor__head">
                        <span>クラス</span>
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
                    title="後処理stage全体を定義する上級者向け設定。クラス別KF JSONとは併用できません。"
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
                    label="キーフレーム間隔"
                    title="Productionポリゴンの目標キーフレーム間隔の指定方法です。"
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
                      label="KF設定JSON"
                      stack
                      title="クラス別のProductionポリゴン目標間隔を持つJSONです。"
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
                          <span>KF間隔</span>
                          <span />
                        </div>
                        <div className="policy-editor__rule policy-editor__rule--default">
                          <TextInput
                            value="その他（未指定）"
                            disabled
                            onChange={() => undefined}
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
                                  keyframeInterval:
                                    postprocess.keyframeInterval ?? 3,
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
  );
}
