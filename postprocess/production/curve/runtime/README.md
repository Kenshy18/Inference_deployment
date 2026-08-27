# Catmull--Rom curve runtime

閉じたuniform Catmull--Rom曲線を、トラック対応点、候補状態、キーフレームDPの
順に最適化するProduction runtimeです。編集点`P[i]`だけを変数とし、Bezier handleは
隣接点から固定の`1/6`式で導出します。自由Bezier fittingは行いません。

## 実行順序

```text
curve/stage.py
  -> curve/engine.py
    -> runtime/fitter.py + runtime/spatial.py
    -> runtime/role_states.py
    -> runtime/multistate_dp.py
      -> runtime/keyframe_dp.py
    -> runtime/metrics.py + runtime/topology.py
    -> curve/storage.py
```

## モジュールの責務

| 場所 | 責務 |
|---|---|
| `model.py` | Catmull--Romから固定Bezier区間への変換とsampling |
| `curve_fit.py` | 曲線全体を対象にした線形fit |
| `fitter.py` | トラック内で意味が持続する対応点をfit |
| `spatial.py` | 各フレームの空間fit、Recall修復、品質記録 |
| `role_states.py` | 時間変形に対応する少数の候補状態 |
| `multistate_dp.py` | 候補状態を含むProduction DP |
| `keyframe_dp.py` | hard minimum-Recall、key penalty、pair refinement |
| `metrics.py` | Recall、IoU、膨張率、時間安定性の集計 |
| `topology.py` | sampled curveの自己交差検査 |
| `native_cpu.py` | OpenCV互換の厳密CPU raster batch |

ポリゴンruntimeから共有するのは、頂点数policy、候補形状の時間的役割、
永続頂点配置の補助演算です。曲線のraster、topology、DP endpoint rendererはこの
ディレクトリが所有し、ポリゴンへ暗黙変換して評価しません。

## 変更時の不変条件

- 点数はトラック単位で固定し、フレームごとに変えない
- `P[i]`の位相対応を維持し、Bezier handleを独立変数にしない
- 全有効フレームでminimum Recallとtopologyを厳密監査する
- キーフレーム間隔はhard上限ではなく、IoUとのsoft trade-offとする
- `postprocess/experiments/`をProduction runtimeからimportしない
