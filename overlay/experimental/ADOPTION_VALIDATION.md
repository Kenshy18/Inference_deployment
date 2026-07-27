# Overlay 3モード採用検証

実施日: 2026-07-27

## 重要な訂正

初回検証に使用した顔検出SQLiteは、RT-DETRへ渡す`[width, height]`を
`[height, width]`としていたため、X座標が圧縮されY座標が伸長していた。構造・時刻
検証は合格していたが、overlayの意味的な位置は不正だった。2026-07-27に
`rtdetr_head_face/preprocessing.py`を修正し、顔検出5290 framesを再実行した。

修正前の顔box中央値は幅121×高さ538、縦横比0.25だった。修正後は幅226×高さ270、
縦横比0.85となり、17135件すべてが1920×1080内、無効box 0件になった。実frame
29、1000、3000でmaskと顔boxの対象位置を確認した。以下の実動画・画質値は、
特記したNVENC速度値を除き修正後の出力へ更新している。

## 判定

次の3モードを採用する。

| `execution_mode` | 実装 | 主用途 | 判定 |
|---|---|---|---|
| `cpu` | OpenCV描画 + libx264 | GPUを使わない基準・fallback | 採用 |
| `nvenc` | OpenCV描画 + NVENC | 通常表示を保った高速encode | 採用 |
| `fast_parallel` | NVDEC + CUDA描画 + NVENC 6分割 | 1080p大量処理 | 条件付き採用 |

`fast_parallel`のフレーム完全性、時刻、音声、分割境界、encode品質には問題を
検出しなかった。ただしOpenCV通常版と描画pixelは同一ではない。フォント、
アンチエイリアス、色変換の見た目を通常版と完全一致させる用途では
`cpu`または`nvenc`を使う。

## 実動画試験

1920x1080、30000/1001 fps、5290 frames、176.509667秒の実入力と、最終mask
4025件、修正済み顔box 17135件を使用した。全モードでmask、顔、ラベルを有効にし、
8 Mbpsを指定した。高速版だけは対応済みの音声stream copyも有効にした。

| モード | 経過時間 | 処理速度 | 実時間比 | 映像frames | 結果 |
|---|---:|---:|---:|---:|---|
| CPU通常 | 41.749秒 | 126.71 fps | 4.23倍 | 5290 | pass |
| NVENC通常* | 38.083秒 | 138.91 fps | 4.64倍 | 5290 | pass |
| 高速6分割 | 3.967秒 | 1333.55 fps | 44.50倍 | 5290 | pass |

* NVENC通常の速度だけは座標修正前の測定。通常CPUと同じOpenCV描画経路なので、
座標修正による速度への影響は小さいが、旧動画は表示比較に使用しない。

高速版はCPU通常の約10.5倍、NVENC通常の約9.6倍だった。実動画の高速出力では
次を確認した。

- 5290 framesが入力範囲と完全一致
- 0秒開始、30000/1001 fps、PTS/DTSが単調かつ一定
- worker範囲`0-881`、`882-1763`、`1764-2645`、`2646-3527`、
  `3528-4408`、`4409-5289`にgap/overlapなし
- 出力frame位置882、1764、2646、3528、4409の全境界がkeyframe
- 全境界のPTS/DTS stepが1 frame分
- 映像とAACを含む全stream decode error 0
- AAC decoded audioは0秒開始、packet/frame timestampにgapなし
- 6 workerが合計4025 masks、17135 facesを描画

## 非GOP境界・途中範囲試験

意図的に分割しにくい条件を作るため、1280x720、30 fps、GOP 60、AAC 48 kHzの
60秒動画を生成した。GOP境界ではないframe 137から1463までを選択し、
1327 framesを6 workerで処理した。

| 項目 | 結果 |
|---|---:|
| 処理速度 | 955.89 fps |
| 出力frames | 1327 / 1327 |
| 出力時間 | 44.233333秒 |
| 最大timestamp誤差 | 0.000000333秒 |
| 5分割境界 | 全て1 frame step、全てkeyframe |
| 全stream decode | error 0 |
| VMAF平均 / 最低 | 98.4647 / 95.7678 |
| float SSIM平均 | 0.999636 |
| PSNR-Y平均 | 48.285 dB |
| 境界VMAF平均 | 98.6067 |
| 非境界VMAF平均 | 98.4620 |

境界VMAFは非境界より0.145高く、分割位置固有の劣化はなかった。音声は元動画の
frame 137相当sampleから、出力映像長相当までをPCMへdecodeして比較し、
両方のMD5が`cc24b63f6f5adda8845f28dea45ab474`で完全一致した。AAC packetが
負のPTSから始まるのは異常ではなく、同じ長さの`Skip Samples` metadataによって
decoder出力は正確に0秒から始まる。

## 画質の分離評価

異なる描画方式同士のpixel差と、encodeによる劣化を混同しないよう分けて測定した。

| 比較 | VMAF平均 | float SSIM | PSNR-Y | 解釈 |
|---|---:|---:|---:|---|
| 通常NVENC vs 通常CPU | 97.7923 | 0.997770 | 45.500 | 通常2モードの差は小さい |
| 高速6分割 vs C++ lossless | 96.9658 | 0.996809 | 43.579 | 高速encode品質は良好 |
| 高速6分割 vs 通常CPU | 87.0899 | 0.984277 | 29.863 | 描画方式差を含むため非同一 |

C++ lossless基準の600 framesでは、境界VMAF 97.3767、非境界VMAF 96.9480で、
境界固有の回帰はなかった。高速版と通常CPU版の比較でも境界VMAF 89.0016、
非境界87.0808であり、低い値はframeずれや結合不良ではない。通常版との差は主に
OpenCV `LINE_AA`とCUDA整数span、OpenCV fontと組み込みASCII font、BGR経路と
BT.709 limited-range YUV直接合成の差である。

したがって品質判定は次のように扱う。

- 元映像のencode品質: pass
- mask/顔件数と時刻上の対応: pass
- 分割境界の画質: pass
- OpenCV通常版とのpixel完全一致: 非対応

## 自動検証

`validate_fast_output.py`は次をfail-fastで確認する。

- 映像frame/packet数
- 映像PTS/DTSの0始まり、単調性、一定間隔
- workerのframe範囲と件数、gap/overlap
- 分割境界のPTS stepとkeyframe
- FFmpegによる全stream decode
- 音声のdecoded開始時刻、AAC preroll metadata、packet/frame連続性
- 任意のlossless基準に対するVMAF、float SSIM、PSNR-Y
- 境界windowと非境界のVMAF差

例:

```bash
python overlay/experimental/validate_fast_output.py \
  --output output/final.mp4 \
  --summary output/benchmark_summary.json \
  --reference output/lossless_reference.mp4 \
  --report output/validation.json \
  --minimum-vmaf-mean 90 \
  --maximum-boundary-vmaf-regression 1
```

## 採用時の制約

- 現在の高速GPU経路はH.264 yuv420p/NV12、CFR、MP4、NVIDIA GPUが対象
- 1080pの推奨は純NVENC 6 worker、4Kは4 worker
- 入力がVFR、別pixel format、別containerの場合は通常モードを使うか再検証する
- 部分出力は元動画の絶対時刻を保持せず、映像・音声とも0秒へrebaseする
- 元入力に独立した`tmcd` timecode trackがある場合のtrack複製は未対応
- 通常CPU/NVENCモードは現時点では映像のみ、高速版はAAC等のstream copyに対応
- GPU負荷、解像度、storageが変わる環境ではworker数を再測定する

実測JSONと動画は
`overlay/output/overlay_visual_comparison_corrected_20260727/`および
`overlay/output/overlay_coordinate_fix_20260727/`以下に保存した。`output/`は
Git管理外なので、恒久的な判定根拠は本書と自動検証器を使用する。
