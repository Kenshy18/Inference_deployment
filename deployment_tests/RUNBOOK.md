# 8時間実行Runbook

## 0. 開始前

1. Git HEAD、worktree差分、時刻、OS、driver、GPU、CPU、RAM、空き容量を記録する。
2. GPUを使う他processがないことを確認する。
3. `gui/scripts/build-windows.sh`でclean commitからreleaseを作り、
   `build-manifest.json`とSHA-256を固定する。NSISをテスト用Windows環境へ新規installし、
   インストール済みexeを起動する。`win-unpacked`は障害切り分け専用とする。
4. `cases.json`を`check_plan.py`で検査する。
5. fixtureをprobeし、期待するcodec、container、resolution、FPS、audio、frame数を
   `work/fixtures-report.json`へ固定する。
6. outputとElectron user-dataはケースごとに新規directoryを割り当てる。
7. resource baselineをGUI起動前、起動後30秒、最初のjob前で取得する。

開始時刻から8時間のhard deadline、7時間35分のnew-job deadline、7時間45分の
forced-finalization deadlineを記録します。

## 1. GUI操作

製品pipelineを開始する操作は全てPlaywrightからインストール済みWindows exeの
実Electron windowへ行います。自動操作portはテスト時だけ明示的に有効化し、通常起動
では無効にします。

1. Sourceの参照ボタンからautomation fixtureを選び、追加ボタンでqueueへ入れる。
2. 簡単／詳細を切り替え、ケースに必要なcontrolをpointerで変更する。
3. 表示値とlocalStorage draftを取得し、意図した設定になったことを記録する。
4. Dry Runボタンを押し、GUIに表示されたresolved command/configを証跡保存する。
5. 実行ボタンを押し、Dry Runとは別job IDであることを確認する。
6. STATUS/LIVE/Consoleを実際に切り替え、screenshotを取得する。
7. output queue項目をclickし、automation modeのopen-output応答を確認する。

localStorageを直接patchする既存matrix方式は、大量ケースの初期値設定には使えますが、
`G00`では使用せず、全基本controlをDOM操作してGUI binding自体を検査します。その他の
ケースも、各ケースの主検証軸に該当するcontrolは必ずDOM操作します。

## 2. 実行中sampling

0.5秒間隔で次をJSONLへ追記します。

- job ID、status、stage、phase進捗、FPS、frame、detections、masks、faces
- Electron heartbeat delay、Playwright evaluate latency、LIVE frame/dropped
- process tree PID、RSS、PSS、CPU、thread、FD
- GPU utilization、VRAM、temperature
- output directory bytes、file count、WAL/SHM、一時directory数

性能gate対象のgolden runでは、別成果物のfull integrity checkや動画decodeを同時に
行いません。

## 3. 中断試験

`R01`は各phaseが`running`になり、進捗が0より大きく100%未満になった時点でGUIの
停止ボタンを押します。

- 30秒以内に`cancelling`から`cancelled`へ遷移する。
- 60秒以内に子process、VRAM、open fileが解放される。
- 未完成MP4/SQLiteをoutput queueへ載せない。
- 同じ入力を右clickで再処理し、新規suffixへ正常完了できる。

overlayは速すぎるため、既存の正常なSQLiteをGUIの「推論を実行しない」「既存
SQLite」設定で読み込み、4K長尺 + CPU overlayを使って停止可能な時間を確保します。

`R02`はphase running中にElectron windowを閉じます。再起動時にjobをcompletedとして
表示しないことを確認し、resumeと最初から再処理をそれぞれ1回試します。強制killは
最後のsubcaseだけにし、その前に通常closeの挙動を保存します。

## 4. ケース終了直後

1. queue、job snapshot、全log、screenshot、resolved configを保存する。
2. run manifestと成果物一覧を保存する。
3. success/cancel/failureごとのcleanup snapshotを60秒後まで取得する。
4. 次job前にGPU/VRAM/process baselineへ戻ったか確認する。
5. P0 failureなら証跡を固定し、同一ケースを1回だけ再実行する。
6. 実測時間をschedulerへ返し、P1/P2を継続できるか再計算する。

## 5. 成果物の読み取り専用検査

GUI job終了後に検証器を実行します。これは製品処理を代替しません。

- SQLite contract、schema signature、integrity、foreign key、capability、row関係
- model別schema signature比較
- frame/cut/geometry/score/track/KF/顔maskの内容検査
- ffprobe packet、PTS/DTS、duration、resolution、audio
- overlay manifestの描画行数とSQLite件数
- worker range、境界frame identity
- threshold未満の顔geometry、mask点滅候補、cut跨ぎ候補
- 一時ファイルとdisk差分

大きいS00 SQLiteのfull `integrity_check`は終了後に1回だけ行います。実行中は
`quick_check`と局所SQLを使い、I/O競合を避けます。

## 6. 目視

検証器が作ったcontact sheetをP0ケースごとに確認し、`visual-signoff.json`へ
`pass`、確認者、コメントを記録します。自動検査だけで位置や見た目を合格にしません。

最低限見るもの:

- S00 V3の開始・中央・終了、G02 V3-lite簡易／詳細
- G04 顔mask 4形態
- M02 4K、M01 portrait/VFR
- S00のworker境界、cut前後

## 7. 終了判定

1. P0/P1/P2のpass/fail/not-runを集計する。
2. 未解決renderer error、stage error、schema差分、cleanup残留を一覧にする。
3. resource slopeと性能baseline差を計算する。
4. `ACCEPTANCE_CRITERIA.md`に従いPASS/CONDITIONAL/FAILを決める。
5. 未実施項目をpassとして数えない。
6. `deployment-report.json`と人間向け`deployment-report.md`を保存する。
7. テスト生成物を削除する前に、failure証跡と再現設定が揃っていることを確認する。

## 禁止事項

- pipelineを直接CLIで起動してGUI試験の代用にしない。
- 既存outputを削除して衝突試験を通したことにしない。
- timeoutを成功扱いしない。
- `S00_v3_120m_single_load`以外でV3の長時間／stressを実行しない。
- schema不一致をmigrationでその場修正して継続しない。
- failure後にproduction codeを修正し、同じ8時間runを継続してPASSにしない。
