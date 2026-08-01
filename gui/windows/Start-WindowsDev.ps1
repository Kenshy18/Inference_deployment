param(
  [string]$Distribution = "Ubuntu-24.04",
  [string]$RepositoryRoot = "/home/kenshin/inference_backend2",
  [string]$RuntimePython = "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
  [string]$DevRoot = "$env:LOCALAPPDATA\MaskPipelineStudioDev"
)

$ErrorActionPreference = "Stop"
$env:Path =
  [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
  [Environment]::GetEnvironmentVariable("Path", "User")

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptRoot "Check-WindowsRuntime.ps1") `
  -Distribution $Distribution `
  -BackendRoot $RepositoryRoot `
  -RuntimePython $RuntimePython

$sourceRoot = "\\wsl.localhost\$Distribution" + ($RepositoryRoot.Replace("/", "\")) + "\gui"
$stageRoot = Join-Path ([System.IO.Path]::GetFullPath($DevRoot)) "source"
New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

$excludedDirectories = @(
  "node_modules",
  "dist",
  "release",
  "output",
  ".runtime-libs",
  "design-proposals"
)
$robocopyArgs = @(
  $sourceRoot,
  $stageRoot,
  "/MIR",
  "/R:2",
  "/W:1",
  "/NFL",
  "/NDL",
  "/NJH",
  "/NJS",
  "/NP",
  "/XD"
) + $excludedDirectories
& robocopy.exe @robocopyArgs | Out-Null
if ($LASTEXITCODE -gt 7) {
  throw "robocopy failed with exit code $LASTEXITCODE."
}

Push-Location $stageRoot
try {
  $lockHash = (Get-FileHash -LiteralPath "package-lock.json" -Algorithm SHA256).Hash
  $stamp = Join-Path $stageRoot "node_modules\.mask-studio-lock-sha256"
  $installedHash = if (Test-Path -LiteralPath $stamp) {
    (Get-Content -LiteralPath $stamp -Raw).Trim()
  } else {
    ""
  }
  if ($installedHash -ne $lockHash) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) {
      throw "npm ci failed with exit code $LASTEXITCODE."
    }
    Set-Content -LiteralPath $stamp -Value $lockHash -Encoding ascii
  }
  $env:INFERENCE_BACKEND_ROOT = $RepositoryRoot
  $env:INFERENCE_RUNTIME_PYTHON = $RuntimePython
  & npm.cmd run dev
  if ($LASTEXITCODE -ne 0) {
    throw "Windows Electron development process failed with exit code $LASTEXITCODE."
  }
} finally {
  Pop-Location
}
