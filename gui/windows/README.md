# Windows release build

GUIソースの正本はWSL上の`inference_backend2/gui`です。Windows側の編集用コピーを
正本にはしません。Windows向けElectron依存関係とexeはWindows上で生成します。

## 通常の流れ

1. WSL側でGUIを変更し、型検査とテストを通してGitへコミットする。
2. WindowsへNode.js 20以降をインストールする。
3. WSLから次を実行する。

```bash
cd /home/kenshin/inference_backend2/gui
./scripts/build-windows.sh
```

PowerShellから直接実行する場合:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  \\wsl.localhost\Ubuntu-24.04\home\kenshin\inference_backend2\gui\windows\Build-Windows.ps1
```

スクリプトはソースだけを`D:\GUI_frontend\build\source`へ同期し、Windows上で
`npm ci`、型検査、単体テスト、NSIS/portableビルドを実行します。`node_modules`、
過去のrelease、出力動画は同期しません。

成果物は`D:\GUI_frontend\release\<version>`へ出力されます。

- `Mask Pipeline Studio-Setup-<version>-x64.exe`
- `Mask Pipeline Studio-Portable-<version>-x64.exe`
- `win-unpacked/`
- `SHA256SUMS.txt`
- `build-manifest.json`

通常起動時に同期やビルドは行いません。GUIソースを変更して新しいexeが必要になった
時だけこの処理を実行します。既存versionを上書きするには`-ReplaceRelease`が必要です。
正式配布では上書きせずversionを上げてください。

## 日常のWindows動作確認

正式なexeを作らず、WSL正本をWindowsの一時開発領域へ差分同期してWindows Electronを
起動できます。

```bash
./scripts/dev-windows.sh
```

既定の一時領域は`%LOCALAPPDATA%\MaskPipelineStudioDev`です。`package-lock.json`が
変わった場合だけ`npm ci`をやり直し、通常のソース変更ではすぐに開発GUIを起動します。

未コミットのリポジトリから正式releaseを作ることは既定で拒否します。開発確認だけに限って
`-AllowDirty`を指定できます。その場合はmanifestにもdirty buildとして記録されます。
