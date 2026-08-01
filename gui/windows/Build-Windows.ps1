param(
  [string]$Distribution = "Ubuntu-24.04",
  [string]$RepositoryRoot = "/home/kenshin/inference_backend2",
  [string]$RuntimePython = "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
  [string]$BuildRoot = "D:\GUI_frontend\build",
  [string]$ReleaseRoot = "D:\GUI_frontend\release",
  [switch]$AllowDirty,
  [switch]$SkipTests,
  [switch]$ReplaceRelease
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-SafeDirectory([string]$Value, [string]$Name) {
  $resolved = [System.IO.Path]::GetFullPath($Value)
  $root = [System.IO.Path]::GetPathRoot($resolved)
  if (-not $root -or $resolved.TrimEnd("\") -eq $root.TrimEnd("\")) {
    throw "$Name must not be a drive root: $resolved"
  }
  return $resolved
}

function Invoke-Checked([string]$Command, [string[]]$Arguments) {
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Command failed with exit code $LASTEXITCODE."
  }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptRoot "Check-WindowsRuntime.ps1") `
  -Distribution $Distribution `
  -BackendRoot $RepositoryRoot `
  -RuntimePython $RuntimePython

$buildRootPath = Assert-SafeDirectory $BuildRoot "BuildRoot"
$releaseRootPath = Assert-SafeDirectory $ReleaseRoot "ReleaseRoot"
$stageRoot = Join-Path $buildRootPath "source"

$commit = (& wsl.exe -d $Distribution -- git -C $RepositoryRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $commit) {
  throw "Could not resolve the canonical WSL Git commit."
}
$statusLines = @(& wsl.exe -d $Distribution -- git -C $RepositoryRoot status --porcelain)
$isDirty = $statusLines.Count -gt 0
if ($isDirty -and -not $AllowDirty) {
  throw "The canonical repository has uncommitted changes. Commit first, or use -AllowDirty for a non-release build."
}

$sourceRoot = "\\wsl.localhost\$Distribution" + ($RepositoryRoot.Replace("/", "\")) + "\gui"
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
  throw "Canonical GUI source was not found: $sourceRoot"
}

if (Test-Path -LiteralPath $stageRoot) {
  Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
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
  Invoke-Checked "npm.cmd" @("ci")
  if (-not $SkipTests) {
    Invoke-Checked "npm.cmd" @("run", "typecheck")
    Invoke-Checked "npm.cmd" @("test", "--", "--run")
  }
  Invoke-Checked "npm.cmd" @("run", "dist:win")

  $version = (& node.exe -p "require('./package.json').version").Trim()
  $packageName = (& node.exe -p "require('./package.json').name").Trim()
  $electronVersion = (& node.exe -p "require('./node_modules/electron/package.json').version").Trim()
  $nodeVersion = (& node.exe --version).Trim()
  $npmVersion = (& npm.cmd --version).Trim()
  $artifactDirectory = Join-Path $releaseRootPath $version
  if (Test-Path -LiteralPath $artifactDirectory) {
    if (-not $ReplaceRelease) {
      throw "Release $version already exists: $artifactDirectory. Bump the version or use -ReplaceRelease explicitly."
    }
    Remove-Item -LiteralPath $artifactDirectory -Recurse -Force
  }
  New-Item -ItemType Directory -Path $artifactDirectory -Force | Out-Null

  $expectedArtifacts = @(
    "Mask Pipeline Studio-Setup-$version-x64.exe",
    "Mask Pipeline Studio-Portable-$version-x64.exe"
  )
  $artifactRecords = @()
  foreach ($name in $expectedArtifacts) {
    $source = Join-Path (Join-Path $stageRoot "release") $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
      throw "Expected Windows artifact was not produced: $source"
    }
    $target = Join-Path $artifactDirectory $name
    Copy-Item -LiteralPath $source -Destination $target
    $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    $signature = Get-AuthenticodeSignature -LiteralPath $target
    $artifactRecords += [pscustomobject]@{
      file = $name
      size_bytes = (Get-Item -LiteralPath $target).Length
      sha256 = $hash
      signature_status = [string]$signature.Status
    }
  }

  $unpackedSource = Join-Path (Join-Path $stageRoot "release") "win-unpacked"
  if (Test-Path -LiteralPath $unpackedSource -PathType Container) {
    Copy-Item -LiteralPath $unpackedSource -Destination $artifactDirectory -Recurse
  }

  $lockHash = (Get-FileHash -LiteralPath (Join-Path $stageRoot "package-lock.json") -Algorithm SHA256).Hash.ToLowerInvariant()
  $manifest = [ordered]@{
    schema_version = 1
    product = "Mask Pipeline Studio"
    package_name = $packageName
    version = $version
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    canonical_repository = $RepositoryRoot
    canonical_commit = $commit
    canonical_repository_dirty = $isDirty
    distribution = $Distribution
    runtime_python = $RuntimePython
    windows_node = $nodeVersion
    windows_npm = $npmVersion
    electron = $electronVersion
    package_lock_sha256 = $lockHash
    artifacts = $artifactRecords
  }
  $manifestPath = Join-Path $artifactDirectory "build-manifest.json"
  $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
  $sumPath = Join-Path $artifactDirectory "SHA256SUMS.txt"
  $artifactRecords | ForEach-Object { "$($_.sha256)  $($_.file)" } | Set-Content -LiteralPath $sumPath -Encoding ascii

  Write-Host "Windows release created: $artifactDirectory" -ForegroundColor Green
  Write-Host "Canonical commit: $commit"
  Write-Host "Test and distribute only artifacts whose hashes match SHA256SUMS.txt."
} finally {
  Pop-Location
}
