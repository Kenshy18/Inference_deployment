# GUI引数カバレッジ

この文書は、`orchestration/config.py`の公開設定を基準に、GUIからどこまで
操作できるかを示します。結論として、利用者が変更してよい公開設定はすべてGUIから
設定できます。ファイル名やstage間の受け渡しを壊す引数だけは、オーケストレーターが
管理します。

GUIでは次の区別を明示します。

- 最終SQLiteへ保存する設定と、確認用overlayだけの描画設定を混在させない
- 適用されないモデル・方式の設定は非表示または無効化する
- `preset`は標準出力、`raw/tracked/final/faces`は必要時だけ追加する互換出力
- codecなど他の選択から一意に決まる値は表示専用項目にせず、自動管理する

表中の区分:

- **簡単**: 普段のジョブで直接操作する項目
- **詳細**: Inspectorを「詳細」に切り替えると操作できる項目
- **入力**: Sourceペインで操作する項目
- **自動**: 他の選択から整合する値を生成する項目
- **内部**: オーケストレーターが所有し、GUIからは変更しない項目

## Orchestration

| 設定 | GUI | 説明 |
| --- | --- | --- |
| `schema_version` | 内部 | 現在は`1` |
| `input_video` | 入力 | 元動画 |
| `output_root` | 入力 | 1ジョブの出力ルート |
| `execution.runtime_python` | 詳細/Runtime | 実行Python |
| `execution.resume` | 詳細 | 完了済みstageの再利用 |

## Inference

| 設定 | GUI | 説明 |
| --- | --- | --- |
| `enabled` | 簡単 | 新規推論または既存SQLite再利用 |
| `input_sqlite` | 入力 | 推論を省略するときのunified SQLite |
| `mode` | 簡単 | 性器、顔、両方 |
| `segmentation_model` | 簡単 | EVA、DINO Cascade、巨大/小型Co-DINO |
| `segmentation_backend` | 簡単 | モデルが対応する推論エンジン |
| `face_model` | 簡単 | 旧RT-DETRまたは新Face DINO v2 |
| `face_backend` | 簡単 | 顔モデルの推論エンジン。現行は新顔=TensorRT、旧顔=PyTorch |
| `face_classes` | 詳細 | SQLiteへ保存する顔出力（Head頭部box / Face顔領域） |
| `face_trt_bundle` | 詳細 | Face DINO v2限定のTRT bundle上書き。旧顔モデルでは非表示 |
| `device` | 詳細 | CUDA device |
| `max_frames` | 詳細 | テスト用フレーム上限 |
| `warmup_frames` | 詳細 | 速度集計から除外する先頭フレーム |
| `face_warmup_iterations` | 詳細 | 顔モデルの空打ち回数 |
| `parallel_models` | 詳細 | 小型Co-DINO + Face DINO v2限定の並行推論 |
| `parallel_model_stagger_seconds` | 詳細 | 並行モデルの開始差 |
| `fast_sqlite` | 詳細 | 耐障害性と引き換えの高速SQLite |
| `extra_args` | 詳細 | 将来の未型付け引数。1トークン1行 |

`face`モードでは性器モデルのキーを設定JSONへ出力しません。
`segmentation`モードでは顔モデルのキーを出力しません。このため、顔のみなどの
組み合わせでも同じGUIから妥当な設定を生成できます。

## Postprocess

| 設定 | GUI | 説明 |
| --- | --- | --- |
| `enabled` | 簡単 | 性器の追跡・整形。顔後処理とは独立 |
| `tracked_sqlite` / `final_sqlite` | 入力 | 後処理を省略する場合の既存SQLite |
| `shape_mode` | 簡単 | 既定のpolygon/ellipse |
| `pipeline_config` | 詳細 | pipeline設定 |
| `class_policy_json` | 詳細 | クラスpolicy |
| `class_postprocess_policy_json` | 詳細 | GUI編集または既存JSON。クラス別shape/keyframe/補完policy |
| `score_min` | 簡単 | 検出スコア下限 |
| `cut_detect` | 簡単 | カット検出 |
| `cut_method` | 詳細 | 高精度またはframe差分 |
| `precompute_cuts_during_inference` | 詳細 | 独立したFFmpeg縮小decodeによるCPU検出をGPU推論と同時実行 |
| `remove_short_tracks_max_frames` | 詳細 | 性器の短命track除去 |
| `keyframe_interval` | 簡単 | 既定キーフレーム間隔 |
| `max_gap` | 詳細 | 補完する欠損フレーム上限 |
| `model_root` / `k2_run_dir` | 詳細 | 楕円K2モデルの場所 |
| `k2_batch_size` | 詳細 | K2 GPU batch |
| `k2_prep_workers` | 詳細 | K2前処理worker |
| `k2_precision` | 詳細 | FP32/FP16 |
| `k2_forward_mode` | 詳細 | `states_only`/`full` |
| `k2_profile_stages` | 詳細 | K2 stage計測 |
| `k2_cudnn_benchmark` | 詳細 | cuDNN benchmark |
| `k2_tf32` | 詳細 | TF32設定 |
| `device` | 詳細 | CPU/auto/CUDA |
| `export_legacy_sqlite` | 詳細 | v1互換SQLiteも出力 |
| `face_mask_target` | 簡単 | 最終SQLiteへ保存する顔privacy mask。なし、顔全体、目元 |
| `eye_mask_shape` | 簡単 | 目元の楕円/長方形 |
| `minimum_eye_confidence` | 詳細 | 目キーポイント採用閾値 |
| `face_tracking_max_gap_frames` | 詳細 | 顔trackの許容gap |
| `face_tracking_high_score_threshold` | 詳細 | 顔track high閾値 |
| `face_tracking_low_score_threshold` | 詳細 | 顔track low閾値 |
| `face_short_track_max_hits` | 詳細 | 短命顔track判定 |
| `face_short_track_keep_score` | 詳細 | 高信頼短命trackの保持閾値 |
| `face_interpolation_max_gap` | 詳細 | 顔trackの補完上限 |
| `extra_args` | 詳細 | 将来の未型付け引数 |

K2設定は以前`extra_args`だけで指定できましたが、現在はオーケストレーターの型付き
設定です。予約引数の二重指定は検証時に拒否されます。

クラス別後処理は「共通」「GUI」「JSON」の3方式です。「GUI」では確定クラス名ごとに
`shape_mode`、`keyframe_interval`、`max_gap`を直接編集します。実行時に
`class_postprocess_policy.json`をジョブ設定フォルダへ保存するため、実行後も適用値を
再現できます。未指定クラスは画面上の既定形状・既定キーフレーム・既定補完上限を
使います。

## Overlay

| 設定 | GUI | 説明 |
| --- | --- | --- |
| `enabled` | 簡単 | overlay生成 |
| `execution_mode` | 簡単 | CPU、NVENC、分割高速 |
| `backend` | 自動 | CPU/NVENC=`python_opencv`、高速=`native` |
| `raw` / `tracked` / `final` / `faces` | 詳細 | presetに加えて生成する旧形式の工程別MP4 |
| `final_include_faces` | 詳細 | 旧式finalへ顔を合成 |
| `presets` | 簡単 | 顔/性器/両方 × 詳細/簡易の6種類 |
| `genital_source` | 詳細 | presetへ描く性器データをAI生マスク/後処理済み最終マスクから選択 |
| `face_mask_target` | 詳細 | 確認動画だけに追加する顔privacy mask。SQLite保存設定とは独立 |
| `eye_mask_shape` | 詳細 | 目元形状 |
| `minimum_eye_confidence` | 詳細 | 目キーポイント閾値 |
| `face_probability_masks` | 詳細 | 顔確率maskの描画 |
| `face_keypoints` | 詳細 | 顔keypointの描画 |
| `face_ellipses` | 詳細 | 顔楕円の描画 |
| `mask_alpha` | 簡単 | mask濃度 |
| `outline_thickness` / `box_thickness` | 詳細 | 線幅 |
| `show_labels` | 詳細 | class/score/trackラベル |
| `codec` | 自動 | execution modeと常に整合 |
| `h264_crf` / `h264_preset` | 詳細 | CPU H.264。presetは高速modeのCPU workerにも適用 |
| `ffmpeg_bin` | 詳細 | FFmpeg実行ファイル上書き |
| `nvenc_cq` / `nvenc_preset` / `nvenc_gpu` | 詳細 | NVENC品質・速度・GPU |
| `workers` / `cpu_workers` | 詳細 | 分割高速modeのworker |
| `copy_audio` / `faststart` | 詳細 | 分割高速modeのmux設定 |
| `target_bitrate_mbps` | 詳細 | 共通bitrate。高速modeでは必須で、空欄なら8 Mbps |
| `start_frame` / `end_frame` | 詳細 | 描画範囲 |
| `progress_every` | 詳細 | 進捗ログ間隔 |
| `extra_args` | 詳細 | 将来の未型付け引数 |

`execution_mode`変更時に`codec`、bitrate、fast専用設定をGUIが整合させます。
相互に矛盾する`fast + h264`のような設定は生成しません。
プリセットと旧形式の工程別出力は排他的ではなく、選択したものをすべて生成します。

## GUI操作にしないCLI引数

以下は機能不足ではなく、stage間の契約を守るため意図的に内部管理します。

| CLI引数 | 所有者 |
| --- | --- |
| inferenceの`--input`、`--output`、`--runtime-python`、`--overwrite` | Orchestration |
| postprocessの`--input-sqlite`、`--input-video`、`--output-dir` | Orchestration |
| `--orchestration-config-json` | Orchestration。再現用設定をSQLiteへ記録 |
| `--precomputed-cuts-json` | Orchestration。推論中カット成果物を自動接続 |
| overlayの`--video`、`--sqlite`、`--face-sqlite`、`--output` | Orchestration |
| overlayの`--manifest`、`--renderer`、`--output-dir`、`--overwrite` | Overlay/Orchestration内部 |
| package結果の入力・出力SQLite | Orchestration。常に1つの最終SQLiteへ統合 |

これらをGUIの自由入力にすると、manifestと実ファイルの不一致、別ジョブのSQLite混入、
最終SQLiteが複数になる問題が起きるため、詳細設定にも出しません。
