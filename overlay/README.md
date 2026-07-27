# overlay

元動画と、`InstanceSegmentation`または`postprocess`が出力したSQLiteから、
確認用オーバーレイ動画を作成します。ほかのリポジトリの内部実装はimportせず、
公開SQLite契約だけに依存します。

## 選択できるオーバーレイ

| `--overlay-type` | 入力SQLite | 内容 |
|---|---|---|
| `raw` | inference SQLite | AIの生出力instance mask |
| `tracked` | `tracked.sqlite` | NMS、カット分割、tracking、短命track削除後 |
| `final` | `predictions.sqlite` | 最終後処理後。顔boxを任意で追加可能 |
| `faces` | inference SQLite | 顔・頭部boxのみ |

`--mode`は`--overlay-type`の短い別名です。

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
--h264-crf 18            CPU H.264品質（小さいほど高品質）
--h264-preset veryfast   libx264 preset
--nvenc-cq 18            通常NVENC品質（小さいほど高品質）
--nvenc-preset p1        NVENC preset（p1最速、p7高品質）
--target-bitrate-mbps N  制約付き目標bitrate
--manifest result.json   入力契約と処理結果を保存
--overwrite              既存出力の置換を許可
```

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
