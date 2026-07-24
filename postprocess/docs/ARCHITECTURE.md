# Architecture

## 境界

このリポジトリは3種類の実行時コードに限定されています。

1. `preprocessing`から`visualization`までの機能モジュール
2. 実装非依存の`contracts`と、組み立てだけを担当する`common`
3. 一気通貫の入口`run_pipeline.py`

`common`はアルゴリズムを実装しません。組み込み名を該当機能のstage classへ
遅延接続し、pipeline設定を読み、成果物契約を検査しながら順番に実行します。
したがって、一気通貫スクリプトが巨大な処理ファイルをimportしている構造では
ありません。

機能モジュールは別の機能モジュールをimportしません。たとえばtrackingは
cut detectionの実装を知りません。`cuts_json`と`nms_jsonl`だけを受け取り
ます。この制約は`tests/test_architecture.py`でAST検査されています。

## 実行の流れ

```text
run_pipeline.py
  -> common.config       pipeline JSONまたは標準構成
  -> common.registry     実装名から1個のstageを生成
  -> common.runner       requires/provides、schema、出力境界を検査
  -> feature/stages.py   機能内のアルゴリズムを実行
  -> contracts           次のstageへ渡す成果物を検証・読み書き
```

各stageには独立した出力ディレクトリが与えられます。入力を上書きせず、
`pipeline_manifest.json`に実行履歴が残るため、途中成果物の確認と再現が
容易です。

## cut detectionを交換する例

新しい`CutDetector`だけを追加する場合は、`cut_detection/`内で`name`と
`detect(jsonl_path, video_path)`を実装し、
`cut_detection.detector.register_cut_detector`へ登録できます。

stage全体を別パッケージから交換する場合:

```python
from dataclasses import dataclass

from contracts import CutList, StageContext, StageResult, write_cut_list


@dataclass(frozen=True)
class NewCutStage:
    threshold: float = 0.8
    name: str = "new_cut"
    requires: frozenset[str] = frozenset({"nms_jsonl"})
    provides: frozenset[str] = frozenset({"cuts_json"})

    def run(self, context: StageContext) -> StageResult:
        frames = (...)  # context.artifacts["nms_jsonl"]を使って検出
        output = context.stage_dir / "cuts.json"
        write_cut_list(output, CutList(tuple(frames), self.name))
        return StageResult({"cuts_json": output})
```

pipeline JSONの`cut_detection`だけを次のように変更します。

```json
{
  "id": "cut_detection",
  "implementation": "my_package.new_cut:NewCutStage",
  "options": {"threshold": 0.8}
}
```

前後は同じ成果物名で接続されるので、NMS、tracking、`run_pipeline.py`の変更は
不要です。

## 新機能を途中へ挿入する例

NMS後にmask cleanupを追加するなら、新stageが`nms_jsonl`を要求し、
`cleaned_jsonl`を提供します。trackingの置換stageは`cleaned_jsonl`と
`cuts_json`を要求します。pipeline JSONの配列へ両方を配置すれば、
runner自体の変更は不要です。

新しい接続は、曖昧な既存成果物名を再利用せず、意味が分かる新しい成果物名で
表現します。

## 検証

- feature間の直接import禁止
- 旧集約パッケージが存在しないこと
- 宣言成果物の不足時に実行前エラーになること
- custom stageを近似とkeyframeの間へ挿入できること
- cut detectionを交換したraw JSONL→最終SQLite E2E
- 全runtime moduleのimportと成果物schema

これらを`make test`で自動検証します。
