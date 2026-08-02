[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$BackendVhd,
  [Parameter(Mandatory=$true)][string]$GuiPortable,
  [Parameter(Mandatory=$true)][string]$Fixture,
  [string]$OutputRoot = "D:\MaskPipelineDeployment\release",
  [string]$ReleaseCommit,
  [string]$AssetCommit = "6f6823927eefc178a55a53c2615c011fc1ce0076",
  [string]$GpuName = "NVIDIA GeForce RTX 5090",
  [string]$DriverVersion = "596.21"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseId = "mask-pipeline-{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")
$target = Join-Path $OutputRoot $releaseId
$payload = Join-Path $target "payload"
if (Test-Path -LiteralPath $target) { throw "Release already exists: $target" }
New-Item -ItemType Directory -Path $payload -Force | Out-Null

if (-not $ReleaseCommit) {
  $ReleaseCommit = (& wsl.exe -d Ubuntu-24.04 -- git -C /home/kenshin/inference_backend2 rev-parse HEAD).Trim()
}
$files = @(
  @{ source=$BackendVhd; target="backend.vhdx"; role="backend" },
  @{ source=$GuiPortable; target="Mask Pipeline Studio.exe"; role="gui" },
  @{ source=$Fixture; target="deployment-smoke.mp4"; role="fixture" }
)
$artifacts = @()
foreach ($record in $files) {
  if (-not (Test-Path -LiteralPath $record.source -PathType Leaf)) { throw "Missing input: $($record.source)" }
  $destination = Join-Path $payload $record.target
  Copy-Item -LiteralPath $record.source -Destination $destination
  $artifacts += [ordered]@{
    role = $record.role
    file = $record.target
    size_bytes = (Get-Item -LiteralPath $destination).Length
    sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$backend = $artifacts | Where-Object role -eq "backend"
$gui = $artifacts | Where-Object role -eq "gui"
$fixtureRecord = $artifacts | Where-Object role -eq "fixture"
$manifest = [ordered]@{
  schema_version = 1
  release_id = $releaseId
  created_at_utc = [DateTime]::UtcNow.ToString("o")
  backend = [ordered]@{ file=$backend.file; release_commit=$ReleaseCommit; asset_commit=$AssetCommit }
  gui = [ordered]@{ file=$gui.file; version="0.1.3" }
  fixture = [ordered]@{ file=$fixtureRecord.file }
  compatibility = [ordered]@{
    windows_build = [Environment]::OSVersion.Version.ToString()
    wsl_version = (& wsl.exe --version | Select-Object -First 1).Trim("`0 ")
    ubuntu = "24.04"
    gpu_name = $GpuName
    driver_version = $DriverVersion
    compute_capability = "12.0"
  }
  artifacts = $artifacts
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $payload "deployment-manifest.json") -Encoding utf8
Copy-Item -LiteralPath (Join-Path $scriptRoot "Deploy-MaskPipeline.ps1") -Destination $target

$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $csc)) { throw "C# compiler was not found: $csc" }
$deployerExe = Join-Path $target "MaskPipelineDeployer.exe"
& $csc /nologo /target:exe /optimize+ "/out:$deployerExe" (Join-Path $scriptRoot "MaskPipelineDeployer.cs")
if ($LASTEXITCODE -ne 0) { throw "Deployer compilation failed" }

$sums = Get-ChildItem -LiteralPath $target -Recurse -File | ForEach-Object {
  "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.FullName.Substring($target.Length + 1)
}
$sums | Set-Content -LiteralPath (Join-Path $target "SHA256SUMS.txt") -Encoding ascii
Write-Host "Deployment release created: $target" -ForegroundColor Green
