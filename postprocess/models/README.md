# Runtime models

この standalone workspace には検証済みの既定 checkpoint が同梱されています。

```text
models/
  k2_v5/
    best_exact.pt
    run_config.json
```

別モデルを利用する場合は同じレイアウトで別 directory に配置し、
`POSTPROCESS_MODEL_ROOT` または `--model-root` で切り替えてください。既定モデルを
上書きせず、モデルごとに config、重み、特徴量統計をまとめることを推奨します。
