# Diagnostics

Productionの出力を変更せず、SQLiteに保存された形状とキーフレーム配置を解析する
読み取り専用ツールです。

時間方向の移動・相似変形・アフィン変形・局所変形とキーフレーム配置の関係を
調べるには、`postprocess`ディレクトリから次を実行します。

```bash
python -m diagnostics.temporal_geometry \
  --source-sqlite /path/to/tracked.sqlite \
  --keyframes-sqlite /path/to/keyframes.sqlite \
  --output /path/to/temporal_geometry.json
```

この診断は動画を開かず、SQLite内のマスク座標だけを読み取ります。
