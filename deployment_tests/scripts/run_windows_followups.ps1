param(
  [string]$Root = "D:\GUI_frontend\deployment-test\work\formal-8h-followups",
  [string]$FixtureSource = "D:\GUI_frontend\deployment-test\work\formal-8h\fixtures",
  [string]$Source = "D:\GUI_frontend\build\source",
  [string]$Exe = "D:\GUI_frontend\qa-install\0.1.3-eea56e3\Mask Pipeline Studio.exe",
  [string]$Repository = "\\wsl.localhost\Ubuntu-24.04\home\kenshin\inference_backend2"
)

$ErrorActionPreference = "Stop"
$Node = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_*\node-v*-win-x64\node.exe" |
  Sort-Object FullName -Descending |
  Select-Object -First 1).FullName
if (-not $Node) { throw "Windows Node LTS was not found" }
if (-not (Test-Path $Exe)) { throw "Installed EXE was not found: $Exe" }

New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path "$Root\fixtures" | Out-Null
Copy-Item "$FixtureSource\*" "$Root\fixtures" -Force
Set-Location $Source
$Failures = @()

function Invoke-Matrix {
  $env:GUI_MATRIX_INSTALLED_EXE = $Exe
  $env:GUI_MATRIX_REPOSITORY_ROOT = $Repository
  $env:GUI_MATRIX_ROOT = "$Root\matrix"
  $env:GUI_MATRIX_CASE = "12_cancel_segmentation,13_cancel_face,14_cancel_postprocess,15_cancel_cpu_overlay,16_ten_item_batch,17_live_off_ab,18_live_on_ab,19_facev2_none"
  New-Item -ItemType Directory -Force -Path "$Root\matrix\fixtures" | Out-Null
  Copy-Item "$FixtureSource\*" "$Root\matrix\fixtures" -Force
  & $Node "scripts\gui-real-matrix.mjs" *>&1 | Out-File "$Root\matrix.log" -Encoding utf8
  if ($LASTEXITCODE -ne 0) { $script:Failures += "matrix:$LASTEXITCODE" }
}

function Invoke-AttachedAudit {
  param(
    [string]$Name,
    [int]$Port,
    [string[]]$Videos,
    [string]$Script,
    [string[]]$ScriptArguments
  )
  $AuditRoot = "$Root\$Name"
  New-Item -ItemType Directory -Force -Path $AuditRoot | Out-Null
  $Profile = "$AuditRoot\user-data"
  $Output = "$AuditRoot\output"
  $Arguments = @("--automation-port=$Port", "--automation-output=$Output", "--user-data-dir=$Profile")
  $Arguments += $Videos | ForEach-Object { "--automation-video=$_" }
  $Process = Start-Process -FilePath $Exe -ArgumentList $Arguments -PassThru
  try {
    & $Node $Script "--endpoint=http://127.0.0.1:$Port" "--report=$AuditRoot\report.json" "--output-root=$Output" @ScriptArguments
    if ($LASTEXITCODE -ne 0) { $script:Failures += "$Name`:$LASTEXITCODE" }
  }
  finally {
    cmd.exe /c "taskkill /PID $($Process.Id) /T /F" 2>$null | Out-Null
  }
}

Invoke-Matrix

Invoke-AttachedAudit `
  -Name "negative-then-valid" `
  -Port 9330 `
  -Videos @("$Root\fixtures\invalid_truncated.mp4", "$Root\fixtures\golden_short.mp4") `
  -Script "scripts\windows-installed-negative-queue.mjs" `
  -ScriptArguments @()

Invoke-AttachedAudit `
  -Name "reprocess-collision" `
  -Port 9331 `
  -Videos @("$Root\fixtures\golden_short.mp4") `
  -Script "scripts\windows-installed-reprocess.mjs" `
  -ScriptArguments @()

$CloseRoot = "$Root\close-relaunch"
New-Item -ItemType Directory -Force -Path $CloseRoot | Out-Null
& $Node "scripts\windows-installed-close-relaunch.mjs" `
  "--exe=$Exe" `
  "--video=$Root\fixtures\real_720p24_45s.mp4" `
  "--output-root=$CloseRoot\output" `
  "--report=$CloseRoot\report.json"
if ($LASTEXITCODE -ne 0) { $Failures += "close-relaunch:$LASTEXITCODE" }

$Summary = [ordered]@{
  schema_version = 1
  completed_at = (Get-Date).ToUniversalTime().ToString("o")
  root = $Root
  failures = $Failures
}
$Summary | ConvertTo-Json -Depth 8 | Set-Content "$Root\followup-summary.json" -Encoding utf8
if ($Failures.Count -gt 0) { exit 1 }
