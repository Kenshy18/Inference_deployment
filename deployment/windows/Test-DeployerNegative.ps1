[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$ReleaseRoot,
  [string]$WorkRoot = "D:\MaskPipelineDeployment\negative-tests",
  [string]$ExistingDistribution = "MaskPipelineQA"
)

$ErrorActionPreference = "Stop"
$ReleaseRoot = [IO.Path]::GetFullPath($ReleaseRoot)
$WorkRoot = [IO.Path]::GetFullPath($WorkRoot)
$sourcePayload = Join-Path $ReleaseRoot "payload"
$deployer = Join-Path $ReleaseRoot "Deploy-MaskPipeline.ps1"
$sourceManifest = Get-Content -LiteralPath (Join-Path $sourcePayload "deployment-manifest.json") -Raw | ConvertFrom-Json
$fixture = Join-Path $sourcePayload $sourceManifest.fixture.file
if (-not (Test-Path -LiteralPath $deployer -PathType Leaf)) { throw "Missing deployer: $deployer" }
if (-not (Test-Path -LiteralPath $fixture -PathType Leaf)) { throw "Missing fixture: $fixture" }

function Write-Utf8Json([string]$Path, [object]$Value) {
  $encoding = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 10) + [Environment]::NewLine, $encoding)
}

function Get-Distros {
  return @(& wsl.exe --list --quiet) | ForEach-Object { $_.Trim("`0 ") } | Where-Object { $_ }
}

function New-NegativePayload([string]$Name, [string]$FailureKind) {
  $caseRoot = Join-Path $WorkRoot $Name
  $payload = Join-Path $caseRoot "payload"
  $install = Join-Path $caseRoot "install"
  New-Item -ItemType Directory -Path $payload -Force | Out-Null
  Copy-Item -LiteralPath $fixture -Destination (Join-Path $payload "invalid.vhdx")
  Copy-Item -LiteralPath $fixture -Destination (Join-Path $payload "gui.exe")
  Copy-Item -LiteralPath $fixture -Destination (Join-Path $payload "fixture.mp4")
  $hash = (Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant()
  $expectedHash = if ($FailureKind -eq "hash") { "0" * 64 } else { $hash }
  $gpuName = if ($FailureKind -eq "gpu") { "Deliberately incompatible GPU" } else { $sourceManifest.compatibility.gpu_name }
  $manifest = [ordered]@{
    schema_version = 1
    release_id = "negative-$Name"
    backend = [ordered]@{ file="invalid.vhdx"; release_commit="negative"; asset_commit="negative" }
    gui = [ordered]@{ file="gui.exe"; version="negative"; commit="negative" }
    fixture = [ordered]@{ file="fixture.mp4" }
    compatibility = [ordered]@{
      gpu_name = $gpuName
      driver_version = $sourceManifest.compatibility.driver_version
    }
    artifacts = @(
      [ordered]@{ role="backend"; file="invalid.vhdx"; sha256=$expectedHash },
      [ordered]@{ role="gui"; file="gui.exe"; sha256=$hash },
      [ordered]@{ role="fixture"; file="fixture.mp4"; sha256=$hash }
    )
  }
  Write-Utf8Json (Join-Path $payload "deployment-manifest.json") $manifest
  return [pscustomobject]@{ Root=$caseRoot; Payload=$payload; Install=$install }
}

function Invoke-ExpectedFailure([string]$Name, [string]$FailureKind, [string]$Distribution) {
  $case = New-NegativePayload $Name $FailureKind
  $arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $deployer,
    "-DistributionName", $Distribution,
    "-InstallRoot", $case.Install,
    "-PayloadRoot", $case.Payload,
    "-AllowNonAdministrator", "-NoLaunch", "-SkipE2E"
  )
  $previousErrorPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $output = @(& powershell.exe @arguments 2>&1)
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorPreference
  $logPath = Join-Path $case.Root "console.log"
  $output | Set-Content -LiteralPath $logPath -Encoding utf8
  $distExists = (Get-Distros) -contains $Distribution
  $distributionSafe = if ($FailureKind -eq "existing") { $distExists } else { -not $distExists }
  $installedVhd = Join-Path $case.Install "backend\ext4.vhdx"
  $partialVhd = "$installedVhd.partial"
  return [pscustomobject]@{
    name = $Name
    exit_code = $exitCode
    rejected = ($exitCode -ne 0)
    distribution_safe = $distributionSafe
    installed_vhd_absent = (-not (Test-Path -LiteralPath $installedVhd))
    partial_vhd_absent = (-not (Test-Path -LiteralPath $partialVhd))
    console_log = $logPath
  }
}

function Test-DefaultPayloadBinding([string]$Suffix) {
  # Do not pass PayloadRoot: this is the normal EXE launch path. An unsafe
  # name stops immediately after the adjacent release payload is resolved.
  $install = Join-Path $WorkRoot "default-payload-binding-$Suffix\install"
  $arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $deployer,
    "-DistributionName", "!", "-InstallRoot", $install,
    "-AllowNonAdministrator", "-NoLaunch"
  )
  $previousErrorPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $output = @(& powershell.exe @arguments 2>&1)
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorPreference
  $text = $output | Out-String
  return [pscustomobject]@{
    exit_code = $exitCode
    passed = ($exitCode -ne 0 -and $text -match "Unsafe distribution name" -and $text -notmatch "Join-Path")
    adjacent_payload_resolved = ($text -notmatch "Deployment manifest is missing")
  }
}

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
$settingsPath = Join-Path $env:APPDATA "mask-pipeline-studio-windows\settings.json"
$settingsBefore = if (Test-Path -LiteralPath $settingsPath) {
  (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash
} else { $null }

$suffix = Get-Date -Format "yyyyMMddHHmmss"
$defaultPayloadBinding = Test-DefaultPayloadBinding $suffix
$results = @(
  Invoke-ExpectedFailure "hash-$suffix" "hash" "MaskPipelineNegativeHash$suffix"
  Invoke-ExpectedFailure "gpu-$suffix" "gpu" "MaskPipelineNegativeGpu$suffix"
  Invoke-ExpectedFailure "existing-$suffix" "existing" $ExistingDistribution
  Invoke-ExpectedFailure "invalid-vhd-$suffix" "invalid-vhd" "MaskPipelineNegativeVhd$suffix"
)
$settingsAfter = if (Test-Path -LiteralPath $settingsPath) {
  (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash
} else { $null }
$settingsUnchanged = $settingsBefore -eq $settingsAfter

$violations = @($results | Where-Object {
  -not $_.rejected -or -not $_.distribution_safe -or
  -not $_.installed_vhd_absent -or -not $_.partial_vhd_absent
})
if (-not $settingsUnchanged) { $violations += "GUI settings changed" }
if (-not $defaultPayloadBinding.passed) { $violations += "Default payload binding failed" }
$summary = [ordered]@{
  schema_version = 1
  status = if ($violations.Count -eq 0) { "passed" } else { "failed" }
  tested_at_utc = [DateTime]::UtcNow.ToString("o")
  release_id = $sourceManifest.release_id
  settings_unchanged = $settingsUnchanged
  default_payload_binding = $defaultPayloadBinding
  results = $results
}
$summaryPath = Join-Path $WorkRoot "negative-tests-$suffix.json"
Write-Utf8Json $summaryPath $summary
$summary | ConvertTo-Json -Depth 8
if ($violations.Count -ne 0) { throw "Negative deployment safety tests failed: $summaryPath" }
