# Runtime models

このruntimeにはProduction polygon候補生成用の軽量point predictorを配置します。

```text
models/
  polygon_point_predictor/
    best.pt
    feature_stats.npz
    run_config.json
```

重いバイナリは配布asset manifestから注入され、Gitには保存しません。旧K2楕円
モデルはProduction経路から撤去済みで、配布物にも含めません。
