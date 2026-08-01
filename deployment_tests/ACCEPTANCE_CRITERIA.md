# 合格基準

## 判定

- **PASS**: 全P0合格、実行したP1合格、未解決blockerなし。
- **CONDITIONAL**: 全P0合格だが、時間切れのP1/P2または非blocker警告が残る。
- **FAIL**: P0が1件でも不合格、schema変動、成果物破損、再現するGUI crash/freeze。

予期しないstage例外や非0終了は、途中まで正しい成果物があってもFAILです。破損入力を
拒否するnegative caseだけは、明確な日本語エラー、未完成成果物の非公開、次queueの
継続を満たせばPASSです。

## GUI

- 実行したinstaller/exeのSHA-256が`build-manifest.json`と一致し、manifestの正本Git
  commitがテスト対象commitと一致する。
- page error、unhandled rejection、console errorが0件。
- bodyと`.app`が常に存在し、black screenが発生しない。
- 操作対象のbutton、select、number、checkbox、tabが反応し、保存後も値が維持される。
- 日本語path、log、tooltip、overlay文字に豆腐・文字化けがない。
- input queueは処理済み項目を保持し、output queueは成功成果物だけを表示する。
- 再処理で過去成果物を失わず、同名出力は`_2`、`_3`となる。
- 進捗はphase内で単調増加し、完了前に100%にならず、phase開始時の全体進捗が逆戻り
  しない。
- running中の進捗／heartbeat更新gapは通常2秒以下。5秒以上のUI無応答はFAIL。
- 100ms heartbeatのp95は250ms以下、p99は1秒以下。model初期化中でも最大3秒以下。
- LIVEはphaseが5秒以上続く場合に3秒以上静止しない。LIVE OFFに戻すとpreview処理が
  停止し、次jobへsubscriberやtimerを残さない。

## pipelineと成果物

- 必須stageが全てexit code 0で、run manifestが`completed`。
- 最終SQLiteは1つだけが公開成果物となる。
- overlayは選択したpresetだけが作られ、manifestとoutput queueが一致する。
- 入力と出力のwidth/height、予定frame範囲、FPS、durationが契約どおり。
- MP4のvideo packet数が処理frame数と一致し、DTSは単調増加。
- worker範囲に欠損・重複がなく、分割境界前後のframe identityが一致する。
- audio copyを選んだケースはstreamが存在し、映像とのduration差が250ms以下。
- 簡易maskはtrack内のKF間で点滅せず、cut、scene、別trackをまたがない。
- 詳細maskはclass、confidence、track IDを表示し、KFの外周だけがhighlightされる。
- 顔詳細はHead box、合格したFace楕円／mask／keypointを表示する。設定閾値未満のHead
  は0件、閾値未満Faceに顔geometryが残る件数は0件。
- CUT表示はcuts tableの該当frameだけに現れる。

## SQLite固定schema

- `PRAGMA integrity_check='ok'`、`foreign_key_check`違反0。
- `public_result_schema_signature`が全ケースで固定値と一致する。
- 顔なし、性器なし、後処理なし、旧顔でもtable/column/index/viewが消えない。
- capabilityの`available`、`row_count`と実table件数が一致する。
- framesは重複せず、動画範囲内。cutsもframe範囲内。
- detection scoreとprobabilityは`[0,1]`。座標にNaN/Infがなく、ellipse半径は正。
- segmentation、classification、face observation、track、segment、keyframeに孤児行が
  ない。
- polygonは3点以上、rectangleは4頂点相当、ellipseはnative parameterを保持する。
- V3とV3-liteでmodel metadata以外の公開schema署名が同一。
- 生出力、分類confidence、最終track/KF、cut、動画・推論metadata、Face V2のHead/
  Face/ellipse/mask/keypoint、選択したprivacy maskが設定に応じて存在する。

## cleanupと耐障害性

- 成功後に`.orchestrating-*`、`.tmp`、`.partial`、未結合segment、fast cache、孤立WAL/
  SHMが残らない。
- GUI停止または終了後60秒以内に子processが0になり、GPU contextが解放される。
- cancelled jobをcompletedとしてoutput queueへ出さない。
- 再起動後、cancelled/failedを成功扱いせず、安全に再処理またはresumeできる。
- job間でprocess数、FD、temporary directory数が累積しない。
- 最終成果物、manifest、必要なlog以外の中間容量がjobごとに蓄積しない。
- disk不足、読取不能、破損動画は入力を壊さず、理解可能なエラーで停止または次queueへ
  継続する。

## メモリ・VRAM

resourceはGUI親processだけでなく全子process treeを合算します。

- warm-up後の同一phaseでRSS/PSSをframe数に回帰し、明確な単調線形増加がない。
- S00後半50%のPSS slopeが128 MiB/hour以下、または前半／後半中央値差が10%以下。
- GPU inference終了後120秒以内にVRAMが開始前baseline + 512 MiB以下へ戻る。
- batch各job終了後のGUI+常駐process PSSが初回終了後baselineから20%を超えて増えない。
- FD増加がbatch 1本あたり1個未満で、終了時baseline + 20以下。
- OOM、swap storm、thermal shutdown、GPU resetが0件。

## 速度

固定golden fixtureを同一設定で測り、過去の有効baselineまたは当日最初のclean runと
比較します。

- warning: stage FPSまたはwall timeがbaselineから10%以上悪化。
- FAIL: 再測定中央値が20%以上悪化し、入力差・温度・他processで説明できない。
- 3回測る短尺A/Bではcoefficient of variation 12%以下。
- LIVE ONによる推論compute FPS低下は8%以下。
- overlay threshold適用、詳細文字、日本語化による同条件FPS低下は5%以下。
- S00の後半25% throughputが前半25%より15%以上低下しない。

絶対値は機種固有なので主判定にしません。ただしこのPCのsanity floorとして、固定
1080p fixtureでV3 20 FPS、V3-lite 100 FPS、Face V2 120 FPSを下回れば、baselineが
なくても調査対象とします。

## 人間による成果物品質

自動検査合格後、contact sheetと短い境界clipを目視します。

- mask/box/ellipse/keypointの位置ずれなし。
- polygonの頂点密度と輪郭品質が現行original後処理相当。
- ellipseの重複領域に穴がない。
- mask色、alpha、文字太さ、日本語class、track IDが読める。
- KF間のちらつき、cut跨ぎ、track跨ぎ、分割境界の瞬断なし。
- 顔privacy maskが選択したnone/rectangle/ellipse/full-faceと一致。

目視未実施のP0成果物が1つでもあれば最終判定はCONDITIONAL止まりです。
