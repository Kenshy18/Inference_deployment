# overlay

元動画と、`InstanceSegmentation`または`postprocess`が出力したSQLiteから、
確認用オーバーレイ動画を作成します。ほかのリポジトリの内部実装はimportせず、
公開SQLite契約だけに依存します。

後処理済みの標準入力は、推論生出力、`tracking_assignments`、最終編集
キーフレーム、顔生出力を同居させた単一のV3 `result.sqlite`です。同じ
ファイルを`raw`、`tracked`、`final`、`faces`の全モードへ渡せます。
`tracked`は追跡参照と生マスクから、`final`はtyped keyframeから、必要範囲の
一時的な毎フレームcacheを生成します。`fast`では6つの描画範囲を別processで
同時復元し、各workerが専用shardを直接読みます。楕円と長方形は点列JSONへ
展開せずtyped parameterのままC++へ渡します。cacheは公開SQLiteへ書き戻しません。
従来の推論のみSQLiteおよびmaskのみSQLiteも読み取り互換として残します。

## 選択できるオーバーレイ

| `--overlay-type` | 入力SQLite | 内容 |
|---|---|---|
| `raw` | inference SQLite | AIの生出力instance mask |
| `tracked` | `tracked.sqlite` | NMS、カット分割、tracking、短命track削除後 |
| `final` | `predictions.sqlite` | 最終後処理後。顔boxを任意で追加可能 |
| `faces` | inference SQLite | 顔・頭部box、楕円、確率mask、keypoint |

`--mode`は`--overlay-type`の短い別名です。

利用者向けの表示形式は、対象3種類×詳細度2種類のpresetで指定できます。

```text
genital-detailed   genital-simple
face-detailed      face-simple
combined-detailed  combined-simple
```

性器を含むpresetでは`--genital-source raw|final`を併用します。詳細な顔表示は
Head box、face moment-maskの点線境界、顔楕円、可視/遮蔽keypointと確信度を
表示します。簡易表示は性器の最終binary maskと、顔楕円・keypointだけです。
左上の全体HUDは表示しません。
性器を含む簡易presetは固定ピンク`RGB(255, 105, 180)`、mask alphaは
既定`0.45`です。詳細版および従来表示のalphaは`0.32`です。必要なら
`--mask-alpha`で明示的に上書きできます。

`genital-source: final`の形状は後処理側で選択した`ellipse`または`polygon`を
そのまま使用します。これは表示presetとは独立した後処理軸です。複数楕円や
複数polygonが重なる場合は、偶奇塗りで相殺せずunionしたbinary maskとして
表示します。

```bash
overlay-render \
  --preset combined-detailed \
  --genital-source final \
  --execution-mode nvenc \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --face-sqlite inference.sqlite \
  --output output/combined_detailed.mp4
```

presetは表示形式を確定する段階のため、現在は`cpu`と`nvenc`が対応します。
`fast`への移植は通常rendererとの表示検証後に行います。

### 顔・目のプライバシーマスク

Face DINO v2のschema-v3入力では、診断用の楕円・keypoint表示とは別に、実際の
モザイク対象領域を半透明塗りで確認できます。既定は`none`なので従来表示を
変更しません。

```bash
# 顔全体: 検出器の正確な顔楕円
overlay-render \
  --preset face-simple \
  --face-mask-target face \
  --video input.mp4 --sqlite inference.sqlite --output face.mp4

# 目: 左右Eye点から求めた回転楕円
overlay-render \
  --preset face-simple \
  --face-mask-target eyes \
  --eye-mask-shape ellipse \
  --video input.mp4 --sqlite inference.sqlite --output eyes.mp4

# 目: より保護範囲の広い回転長方形
overlay-render \
  --preset face-simple \
  --face-mask-target eyes \
  --eye-mask-shape rectangle \
  --video input.mp4 --sqlite inference.sqlite --output eyes_box.mp4
```

目マスクは、信頼度`--minimum-eye-confidence`以上の2つのvalidなEye点が顔楕円
に対して幾何的に妥当なら、その中点・距離・傾きを使います。欠落、低信頼、
不自然な間隔や向きの場合は、顔楕円とvalidな顔keypointから上顔面の安全な
アイバンドへフォールバックします。領域はEye点そのものだけでなく、まぶた、
眉、目尻まで含む余白を持ちます。詳細表示では導出方法もラベル表示します。
`--no-face-keypoints`や`--no-face-ellipses`は診断描画だけを隠し、マスク導出に
必要な値の読み込みは止めません。

派生マスクを後続のモザイク処理へ渡す場合は、推論SQLiteを変更せず、別の
監査可能なSQLiteへ出力します。

```bash
overlay-export-face-masks \
  --sqlite inference.sqlite \
  --output eyes.sqlite \
  --target eyes \
  --eye-shape ellipse
```

出力は既存readerが読める`masks(frame, track_id, polygons, shape_type, label)`
契約を満たし、`source_observation_id`、`derivation`、`confidence`も保持します。
出力先が既に存在する場合は`--overwrite`が必要です。入力と出力に同じSQLiteを
指定することはできず、一時ファイル完成後のatomic replaceで確定します。

## 選択できる実行方式

| `--execution-mode` | 描画・encode経路 | 用途 |
|---|---|---|
| `cpu` | Python/OpenCV描画＋libx264 | GPUを使わない通常モード |
| `nvenc` | Python/OpenCV描画＋NVENC | 通常表示を保ったGPU encode |
| `fast` | C++/libav、NVDEC、CUDA描画、分割NVENC | 対応入力の最大スループット |

`cpu`と`nvenc`の描画は同じです。`fast`は速度優先のYUV/NV12直接描画なので、
マスク位置、フレーム対応、時刻は同じですが、フォント、アンチエイリアス、
境界pixelは通常モードと完全一致しません。

## セットアップ

```bash
cd /home/kenshin/inference_backend2/overlay
python3 -m venv .venv
.venv/bin/pip install -e .
```

高速モードは、この環境に用意したFFmpeg/CUDA runtimeを使ってnative rendererを
buildします。

```bash
./native/build.sh
```

## 基本的な使い方

CPUでAI生出力を描画:

```bash
overlay-render \
  --execution-mode cpu \
  --overlay-type raw \
  --video input.mp4 \
  --sqlite inference.sqlite \
  --output output/raw.mp4 \
  --manifest output/raw.json
```

NVENCでtracking後を描画:

```bash
overlay-render \
  --execution-mode nvenc \
  --overlay-type tracked \
  --video input.mp4 \
  --sqlite 04_tracking/tracked.sqlite \
  --output output/tracked.mp4
```

CPUで最終maskと顔boxを描画:

```bash
overlay-render \
  --execution-mode cpu \
  --overlay-type final \
  --video input.mp4 \
  --sqlite 07_mask_gap_fill/predictions.sqlite \
  --include-faces \
  --face-sqlite inference.sqlite \
  --output output/final_with_faces.mp4
```

顔boxだけを描画:

```bash
overlay-render \
  --execution-mode nvenc \
  --overlay-type faces \
  --video input.mp4 \
  --sqlite inference.sqlite \
  --output output/faces.mp4
```

高速モード:

```bash
overlay-render \
  --execution-mode fast \
  --overlay-type final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output output/final_fast.mp4 \
  --target-bitrate-mbps 8 \
  --workers 6 \
  --cpu-workers 0 \
  --nvenc-preset p1 \
  --copy-audio \
  --faststart
```

実測上の1080p既定構成はNVENC 6 workerです。CPUとNVENCを混成する場合は
`--workers 6 --cpu-workers 3`でCPU 3＋NVENC 3にできます。最適値は解像度、
入力codec、ストレージ、同時GPU負荷に依存します。

## 主な設定

```text
--start-frame N          開始フレーム
--end-frame N            終了フレーム（含む）
--mask-alpha 0.32        mask塗りの透明度
--outline-thickness 2    mask輪郭の太さ
--box-thickness 2        boxの太さ
--no-labels              class、score、track IDを非表示
--no-face-probability-masks  顔の確率mask塗りを省略
--no-face-keypoints      顔keypointを省略
--no-face-ellipses       顔楕円の代わりに検出boxを描画
--face-mask-target       none / face / eyes
--eye-mask-shape         ellipse / rectangle
--minimum-eye-confidence Eye点を直接使う最低confidence（既定0.35）
--h264-crf 18            CPU H.264品質（小さいほど高品質）
--h264-preset veryfast   libx264 preset
--nvenc-cq 18            通常NVENC品質（小さいほど高品質）
--nvenc-preset p1        NVENC preset（p1最速、p7高品質）
--target-bitrate-mbps N  制約付き目標bitrate
--manifest result.json   入力契約と処理結果を保存
--overwrite              既存出力の置換を許可
```

Face DINO v2の通常rendererで最も重い要素は、フレームごとの顔確率maskの
拡大・alpha合成です。輪郭・位置確認を主目的とし、最大速度を優先する場合は
`--no-face-probability-masks`を使うと、顔楕円とkeypointを残したまま
確率mask塗りだけを省略できます。manifestの`face_components`に実際の設定が
記録されます。

`fast`は比較可能な容量と高速分割を保証するため
`--target-bitrate-mbps`が必須です。`--copy-audio`と`--faststart`は現在
`fast`だけが対応します。通常モードの出力は映像のみです。

旧CLIの`--codec mp4v`、`--codec h264`、`--codec h264_nvenc`も互換目的で
通常モードに残しています。新規実行では`--execution-mode`を使用してください。

## 出力

出力先は自動作成されます。`--manifest`を指定すると、実行方式、overlay種別、
入力SQLite role、フレーム範囲、描画件数、encode設定をJSONに記録します。
途中ファイルは隠し作業ディレクトリで作成し、完成後に出力へatomicに移動します。
高速処理が失敗した場合だけ、原因調査用のworkerログを作業ディレクトリに残します。
高速runnerは入力packet PTSを一度だけ索引化し、各workerを正確なkeyframeへ
seekします。SQLiteのframe番号はdecode順で付けるため、stream copyやconcat由来の
PTS gapがあってもframeを飛ばしません。要求範囲と実出力が1枚でも違う場合は
処理を失敗させます。索引時間と非均一PTS間隔数はmanifestの
`summary.source_frame_index`に記録されます。
V3の復元時間、論理mask件数、shard容量は
`summary.keyframe_materialization`に記録され、合計FPSには復元時間も含まれます。

orchestrationでは4種類を任意の組み合わせで一括生成できます。設定例は
[`../orchestration/configs/production.example.json`](../orchestration/configs/production.example.json)
を参照してください。

## 対応範囲と検証

高速経路は現在、H.264 yuv420p/NV12、CFR、MP4、NVIDIA GPUを対象とします。
VFR、別pixel format、別container、通常版とのpixel完全一致が必要な入力では
`cpu`または`nvenc`を使用してください。

```bash
make test
./native/run_tests.sh
```

高速出力のフレーム、PTS/DTS、分割境界、音声、decode完全性を調べるには:

```bash
overlay-validate \
  --output output/final_fast.mp4 \
  --summary output/final_fast.json \
  --report output/validation.json
```

入力SQLiteの詳細は[CONTRACT.md](CONTRACT.md)、採用時の品質検証は
[docs/ADOPTION_VALIDATION.md](docs/ADOPTION_VALIDATION.md)、速度実測は
[docs/PERFORMANCE.md](docs/PERFORMANCE.md)、native内部は
[native/README.md](native/README.md)を参照してください。
