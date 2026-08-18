[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$RepositoryRoot,
  [string]$SourceDistribution = "Ubuntu-24.04",
  [ValidateSet("core", "all")][string]$Profile = "all",
  [string]$WorkRoot = "D:\MaskPipelineDeployment\work",
  [string]$OutputRoot = "D:\MaskPipelineDeployment\release",
  [string]$BuildDistribution,
  [switch]$KeepBuildDistribution,
  [switch]$KeepWork
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($KeepBuildDistribution -and -not $KeepWork) {
  throw "-KeepBuildDistribution requires -KeepWork"
}

function Invoke-Checked([string]$Command, [string[]]$Arguments) {
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Command failed with exit code $LASTEXITCODE"
  }
}

function Get-Distros {
  return @(& wsl.exe --list --quiet) |
    ForEach-Object { $_.Trim("`0 ") } |
    Where-Object { $_ }
}

function Wait-DistroStopped([string]$Distribution) {
  for ($attempt = 0; $attempt -lt 60; $attempt++) {
    $running = @(& wsl.exe --list --running --quiet) |
      ForEach-Object { $_.Trim("`0 ") } |
      Where-Object { $_ }
    if ($running -notcontains $Distribution) { return }
    Start-Sleep -Milliseconds 500
  }
  throw "WSL distribution did not stop: $Distribution"
}

function Export-WslArchive([string]$Distribution, [string]$Destination) {
  if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination -Force
  }
  Invoke-Checked "wsl.exe" @("--export", $Distribution, $Destination)
}

function Convert-ToWslPath([string]$Distribution, [string]$WindowsPath) {
  # Windows PowerShell 5 does not quote native arguments that contain no
  # spaces. wsl.exe then consumes single backslashes as Linux shell escapes
  # (D:\dir becomes D:dir). Double them explicitly before crossing the WSL
  # command-line boundary.
  $escaped = $WindowsPath.Replace("\", "\\")
  $output = @(& wsl.exe -d $Distribution --cd / -- wslpath -u -- $escaped)
  if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) {
    throw "Could not convert Windows path for WSL: $WindowsPath"
  }
  $converted = ($output -join "`n").Trim()
  if (-not $converted) { throw "WSL returned an empty path for: $WindowsPath" }
  return $converted
}

function Restore-WslInterop([string]$Distribution, [string]$Root) {
  & wsl.exe -d $Distribution -u root --cd / -- `
    "$Root/deployment/phase3/restore_wsl_interop.sh"
  if ($LASTEXITCODE -ne 0) { throw "Could not restore WSLInterop" }
}

$commit = (& wsl.exe -d $SourceDistribution --cd / -- git -C $RepositoryRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $commit) { throw "Could not resolve source commit" }
& wsl.exe -d $SourceDistribution --cd / -- git -C $RepositoryRoot diff-index --quiet HEAD --
if ($LASTEXITCODE -eq 1) { throw "Release builds require committed tracked files" }
if ($LASTEXITCODE -gt 1) { throw "Could not verify the source worktree" }

$releaseStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$releaseId = "mask-pipeline-$releaseStamp-$($commit.Substring(0, 8))"
if (-not $BuildDistribution) { $BuildDistribution = "MaskPipelineBuild-$releaseStamp" }
if ($BuildDistribution -notmatch '^[A-Za-z0-9_.-]+$') {
  throw "Unsafe build distribution name: $BuildDistribution"
}
if ((Get-Distros) -contains $BuildDistribution) {
  throw "Build distribution already exists: $BuildDistribution"
}

$workDirectory = Join-Path $WorkRoot $releaseId
$stageDirectory = Join-Path $workDirectory "stage"
$distroDirectory = Join-Path $workDirectory "build-distro"
$imageDirectory = Join-Path $workDirectory "image"
$guiBuildDirectory = Join-Path $workDirectory "gui-build"
$guiReleaseDirectory = Join-Path $workDirectory "gui-release"
if (Test-Path -LiteralPath $workDirectory) { throw "Work directory already exists: $workDirectory" }
New-Item -ItemType Directory -Path $workDirectory, $imageDirectory -Force | Out-Null

try {
  Write-Host "[1/8] Staging the clean Git commit, assets and runtime..." -ForegroundColor Cyan
  $stageWsl = Convert-ToWslPath $SourceDistribution $stageDirectory
  Invoke-Checked "wsl.exe" @(
    "-d", $SourceDistribution, "-u", "kenshin", "--cd", $RepositoryRoot, "--",
    "./deployment/stage_release.sh", "--stage-root", $stageWsl, "--profile", $Profile
  )

  Write-Host "[2/8] Building the portable Windows GUI from the same commit..." -ForegroundColor Cyan
  $guiBuilder = "\\wsl.localhost\$SourceDistribution" +
    ($RepositoryRoot.Replace("/", "\")) + "\gui\windows\Build-Windows.ps1"
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $guiBuilder `
    -Distribution $SourceDistribution `
    -RepositoryRoot $RepositoryRoot `
    -BuildRoot $guiBuildDirectory `
    -ReleaseRoot $guiReleaseDirectory `
    -ReplaceRelease
  if ($LASTEXITCODE -ne 0) { throw "Windows GUI build failed" }
  $portableGui = @(Get-ChildItem -LiteralPath $guiReleaseDirectory -Recurse -File `
    -Filter "Mask Pipeline Studio-Portable-*-x64.exe")
  if ($portableGui.Count -ne 1) {
    throw "Expected one portable GUI, found $($portableGui.Count)"
  }
  $guiBuildManifestPath = @(Get-ChildItem -LiteralPath $guiReleaseDirectory -Recurse -File `
    -Filter "build-manifest.json")
  if ($guiBuildManifestPath.Count -ne 1) { throw "Expected one GUI build manifest" }
  $guiBuildManifest = Get-Content -LiteralPath $guiBuildManifestPath[0].FullName -Raw | ConvertFrom-Json
  if ($guiBuildManifest.canonical_commit -ne $commit) {
    throw "GUI build does not match source commit $commit"
  }

  Write-Host "[3/8] Creating an isolated clean Ubuntu 24.04 build distribution..." -ForegroundColor Cyan
  Invoke-Checked "wsl.exe" @(
    "--install", "Ubuntu-24.04", "--name", $BuildDistribution,
    "--location", $distroDirectory, "--version", "2", "--no-launch"
  )
  # Ubuntu's first boot enables systemd. systemd-binfmt can remove the shared
  # WSLInterop registration when this temporary distro stops, so install the
  # production wsl.conf before doing any image work, restart once, and restore
  # the source distro's registration defensively.
  Invoke-Checked "wsl.exe" @(
    "-d", $BuildDistribution, "-u", "root", "--cd", "/", "--",
    "install", "-m", "0644", "$stageWsl/wsl.conf", "/etc/wsl.conf"
  )
  Invoke-Checked "wsl.exe" @("--terminate", $BuildDistribution)
  Restore-WslInterop $SourceDistribution $RepositoryRoot

  Write-Host "[4/8] Constructing and validating the production image..." -ForegroundColor Cyan
  $buildStageWsl = $stageWsl
  Invoke-Checked "wsl.exe" @(
    "-d", $BuildDistribution, "-u", "root", "--cd", "/", "--",
    "bash", "$buildStageWsl/bootstrap_image.sh",
    "--stage-root", $buildStageWsl,
    "--release-commit", $commit,
    "--asset-commit", $commit,
    "--profile", $Profile
  )
  $fixture = Join-Path $stageDirectory "deployment-smoke.mp4"
  $fixtureWsl = "$buildStageWsl/deployment-smoke.mp4"
  Invoke-Checked "wsl.exe" @(
    "-d", $BuildDistribution, "-u", "root", "--cd", "/", "--",
    "cp", "/opt/mask-pipeline/fixtures/deployment-smoke.mp4", $fixtureWsl
  )

  Write-Host "[5/8] Exporting the validated WSL image..." -ForegroundColor Cyan
  Invoke-Checked "wsl.exe" @("--terminate", $BuildDistribution)
  Wait-DistroStopped $BuildDistribution
  Start-Sleep -Seconds 2
  $backendArchive = Join-Path $imageDirectory "backend.tar"
  Export-WslArchive $BuildDistribution $backendArchive

  Write-Host "[6/8] Building the transactional Windows deployer..." -ForegroundColor Cyan
  $deployerBuilder = "\\wsl.localhost\$SourceDistribution" +
    ($RepositoryRoot.Replace("/", "\")) + "\deployment\windows\Build-Deployer.ps1"
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $deployerBuilder `
    -BackendArchive $backendArchive `
    -GuiPortable $portableGui[0].FullName `
    -Fixture $fixture `
    -OutputRoot $OutputRoot `
    -ReleaseId $releaseId `
    -ReleaseCommit $commit `
    -GuiCommit $commit `
    -DeployerCommit $commit `
    -AssetCommit $commit `
    -Profile $Profile `
    -GuiVersion $guiBuildManifest.version
  if ($LASTEXITCODE -ne 0) { throw "Deployer build failed" }

  Write-Host "[7/8] Verifying release payload hashes..." -ForegroundColor Cyan
  $releaseDirectory = Join-Path $OutputRoot $releaseId
  $manifest = Get-Content -LiteralPath (Join-Path $releaseDirectory "payload\deployment-manifest.json") `
    -Raw | ConvertFrom-Json
  if ($manifest.backend.release_commit -ne $commit -or $manifest.gui.commit -ne $commit) {
    throw "Release manifest does not match source commit $commit"
  }
  if ($manifest.profile -ne $Profile) {
    throw "Release manifest profile does not match build profile $Profile"
  }
  if ($manifest.schema_version -ne 3 -or -not $manifest.installation.default_distribution) {
    throw "Release manifest does not contain the side-by-side installation contract"
  }
  foreach ($artifact in $manifest.artifacts) {
    $path = Join-Path (Join-Path $releaseDirectory "payload") $artifact.file
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $artifact.sha256) { throw "Release hash mismatch: $path" }
  }

  Write-Host "[8/8] Recording the candidate release..." -ForegroundColor Cyan
  $deployerPath = Join-Path $releaseDirectory $manifest.installation.deployer_filename
  if (-not (Test-Path -LiteralPath $deployerPath -PathType Leaf)) {
    throw "Versioned deployer was not created: $deployerPath"
  }
  $report = [ordered]@{
    schema_version = 1
    status = "candidate"
    release_id = $releaseId
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    source_distribution = $SourceDistribution
    source_root = $RepositoryRoot
    source_commit = $commit
    profile = $Profile
    build_distribution = $BuildDistribution
    release_directory = $releaseDirectory
    deployer = $deployerPath
    default_distribution = $manifest.installation.default_distribution
    backend_format = "wsl-tar"
    backend_archive_sha256 = (Get-FileHash -LiteralPath $backendArchive -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  $report | ConvertTo-Json -Depth 5 | Set-Content `
    -LiteralPath (Join-Path $releaseDirectory "build-report.json") -Encoding utf8
  Write-Host "Candidate release created: $releaseDirectory" -ForegroundColor Green
} finally {
  if (-not $KeepBuildDistribution -and ((Get-Distros) -contains $BuildDistribution)) {
    & wsl.exe --terminate $BuildDistribution 2>$null | Out-Null
    & wsl.exe --unregister $BuildDistribution 2>$null | Out-Null
  }
  Restore-WslInterop $SourceDistribution $RepositoryRoot
  if (-not $KeepWork -and (Test-Path -LiteralPath $workDirectory)) {
    $workWsl = Convert-ToWslPath $SourceDistribution $workDirectory
    $workRootWsl = Convert-ToWslPath $SourceDistribution $WorkRoot
    & wsl.exe -d $SourceDistribution -u kenshin --cd / -- `
      "$RepositoryRoot/deployment/cleanup_release_work.sh" $workWsl $workRootWsl
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not clean release work: $workDirectory" }
  }
}
