# Instance topology problem set (2026-08-12)

Production修正前に、次の異なる問題を混同せず目視合意するための問題集です。

1. `cross_instance_duplicate`: 同一フレームに別検出として存在する重複。既存NMSの対象。
2. `intra_instance_*_island`: 1検出マスク内の独立した前景島。既存NMSの対象外で、連結成分cleanupが必要。
3. `hole_not_island`: 外周内の背景穴。島として削除するのではなく、holeとして保持または明示的に埋める判断が必要。

生成画像はSQLiteのマスク座標を黒いキャンバスへ描いただけで、動画は一切decodeしません。正確な照合キーは、画像とmanifestに記載された `run_key + zero-based frame + detection_id` です。29.97fpsのタイムコード表示には差異があり得るため、TC文字列よりframeを優先します。

```bash
cd /home/kenshin/inference_backend2
PYTHONPATH=postprocess \
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  postprocess/experimental/instance_topology_problem_set_20260812/build_problem_set.py
```

このフォルダは問題把握専用であり、Productionコード、SQLiteスキーマ、モデル出力を変更しません。

追加の幾何・時間整合性監査も動画を開かず、SQLiteだけを読みます。

```bash
/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  postprocess/experimental/instance_topology_problem_set_20260812/audit_additional_failure_modes.py
```

結果は `output/instance_topology_problem_set_20260812/additional_audit.json` に出力されます。
