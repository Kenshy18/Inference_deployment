import type {
  OverlayExecutionMode,
  OverlayPreset,
} from "../../../shared/types";
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
import { OVERLAY_PRESETS, lines, parseLines } from "./shared";
import type { InspectorSectionProps } from "./types";


export function OverlaySection({
  draft,
  busy,
  open,
  advanced,
  onToggle,
  actions,
}: InspectorSectionProps) {
  const { inference, overlay } = draft;
  const usesFaces = inference.mode !== "segmentation";
  const usesSegmentation = inference.mode !== "face";
  const usesRichFaces = usesFaces && inference.faceModel === "face_dino_v2";
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
  const togglePreset = (preset: OverlayPreset, checked: boolean) => {
    if (!checked && overlay.presets.length === 1 && !legacyOverlaySelected) {
      return;
    }
    actions.overlay({
      presets: checked
        ? [...overlay.presets.filter((item) => item !== preset), preset]
        : overlay.presets.filter((item) => item !== preset),
    });
  };

  return (
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
  );
}
