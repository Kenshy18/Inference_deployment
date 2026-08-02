[CmdletBinding()]
param(
  [string]$DistributionName = "MaskPipelineProduction",
  [string]$InstallRoot = "$env:LOCALAPPDATA\MaskPipeline",
  [string]$PayloadRoot,
  [switch]$SkipE2E,
  [switch]$NoLaunch,
  [switch]$KeepFailedInstall,
  [switch]$AllowNonAdministrator,
  [switch]$AllowCompatibilityMismatch
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($PayloadRoot)) {
  # Windows PowerShell 5.1 can evaluate parameter default expressions before
  # PSScriptRoot is populated. Resolve the adjacent payload in the body.
  $PayloadRoot = Join-Path $scriptDirectory "payload"
}
$BackendRoot = "/home/kenshin/inference_backend2"
$RuntimePython = "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10"
$createdDistribution = $false
$settingsBackup = $null
$guiBackup = $null
$installedVhd = $null
$logPath = $null

function Assert-SafeDirectory([string]$Value, [string]$Name) {
  $resolved = [IO.Path]::GetFullPath($Value)
  $root = [IO.Path]::GetPathRoot($resolved)
  if (-not $root -or $resolved.TrimEnd("\") -eq $root.TrimEnd("\")) {
    throw "$Name must not be a drive root: $resolved"
  }
  return $resolved
}

function Invoke-Checked([string]$Command, [string[]]$Arguments) {
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Command failed with exit code $LASTEXITCODE"
  }
}

function Get-Distros {
  return @(& wsl.exe --list --quiet) | ForEach-Object { $_.Trim("`0 ") } | Where-Object { $_ }
}

function Assert-Hash([string]$Path, [string]$Expected) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Payload file is missing: $Path"
  }
  $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -ne $Expected.ToLowerInvariant()) {
    throw "SHA256 mismatch: $Path expected=$Expected actual=$actual"
  }
}

function Write-Utf8Json([string]$Path, [object]$Value, [int]$Depth = 5) {
  $json = $Value | ConvertTo-Json -Depth $Depth
  $encoding = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, $encoding)
}

function Write-Settings([string]$Distro) {
  $settingsDirectory = Join-Path $env:APPDATA "mask-pipeline-studio-windows"
  $settingsPath = Join-Path $settingsDirectory "settings.json"
  New-Item -ItemType Directory -Path $settingsDirectory -Force | Out-Null
  if (Test-Path -LiteralPath $settingsPath) {
    $script:settingsBackup = Join-Path $InstallRoot "rollback\settings.json"
    New-Item -ItemType Directory -Path (Split-Path $script:settingsBackup) -Force | Out-Null
    Copy-Item -LiteralPath $settingsPath -Destination $script:settingsBackup -Force
  }
  $settings = [ordered]@{
    backendMode = "wsl"
    backendRoot = $BackendRoot
    runtimePython = $RuntimePython
    wslDistro = $Distro
  }
  Write-Utf8Json $settingsPath $settings
  return $settingsPath
}

function Restore-Settings([string]$SettingsPath) {
  if ($script:settingsBackup -and (Test-Path -LiteralPath $script:settingsBackup)) {
    Copy-Item -LiteralPath $script:settingsBackup -Destination $SettingsPath -Force
  } elseif (Test-Path -LiteralPath $SettingsPath) {
    Remove-Item -LiteralPath $SettingsPath -Force
  }
}

function New-Shortcut([string]$Target, [string]$Path) {
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($Path)
  $shortcut.TargetPath = $Target
  $shortcut.WorkingDirectory = Split-Path $Target
  $shortcut.Save()
}

$InstallRoot = Assert-SafeDirectory $InstallRoot "InstallRoot"
$PayloadRoot = Assert-SafeDirectory $PayloadRoot "PayloadRoot"
$manifestPath = Join-Path $PayloadRoot "deployment-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
  throw "Deployment manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1) { throw "Unsupported deployment manifest" }
if ($DistributionName -notmatch '^[A-Za-z0-9_.-]+$') { throw "Unsafe distribution name" }

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
$logPath = Join-Path $InstallRoot ("deploy-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -LiteralPath $logPath -Force | Out-Null
$settingsPath = $null
try {
  Write-Host "[1/8] Verifying deployment payload..." -ForegroundColor Cyan
  foreach ($artifact in $manifest.artifacts) {
    Assert-Hash (Join-Path $PayloadRoot $artifact.file) $artifact.sha256
  }

  Write-Host "[2/8] Checking WSL and GPU compatibility..." -ForegroundColor Cyan
  if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw "WSL is not installed" }
  Invoke-Checked "wsl.exe" @("--status")
  $gpuProbe = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  $gpuProbePath = if ($gpuProbe) { $gpuProbe.Source } else { $null }
  if (-not $gpuProbe) {
    $candidate = Join-Path $env:SystemRoot "System32\DriverStore\FileRepository"
    $gpuProbe = Get-ChildItem $candidate -Recurse -Filter nvidia-smi.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($gpuProbe) { $gpuProbePath = $gpuProbe.FullName }
  }
  if (-not $gpuProbe) { throw "NVIDIA driver utility nvidia-smi.exe was not found" }
  $gpuName = (& $gpuProbePath --query-gpu=name --format=csv,noheader).Trim()
  $driverVersion = (& $gpuProbePath --query-gpu=driver_version --format=csv,noheader).Trim()
  if ($LASTEXITCODE -ne 0) { throw "NVIDIA GPU probe failed" }
  if (-not $AllowCompatibilityMismatch) {
    if ($gpuName -ne $manifest.compatibility.gpu_name) {
      throw "GPU mismatch: validated='$($manifest.compatibility.gpu_name)' current='$gpuName'"
    }
    if ($driverVersion -ne $manifest.compatibility.driver_version) {
      throw "NVIDIA driver mismatch: validated='$($manifest.compatibility.driver_version)' current='$driverVersion'"
    }
  }

  Write-Host "[3/8] Reserving a new, isolated WSL distribution..." -ForegroundColor Cyan
  if ((Get-Distros) -contains $DistributionName) {
    throw "WSL distribution '$DistributionName' already exists. Existing distributions are never overwritten."
  }
  $backendDirectory = Join-Path $InstallRoot "backend"
  if (Test-Path -LiteralPath $backendDirectory) {
    $entries = @(Get-ChildItem -LiteralPath $backendDirectory -Force)
    if ($entries.Count -gt 0) { throw "Backend install directory is not empty: $backendDirectory" }
  }
  New-Item -ItemType Directory -Path $backendDirectory -Force | Out-Null
  $sourceVhd = Join-Path $PayloadRoot $manifest.backend.file
  $script:installedVhd = Join-Path $backendDirectory "ext4.vhdx"
  $temporaryVhd = "$installedVhd.partial"
  Copy-Item -LiteralPath $sourceVhd -Destination $temporaryVhd
  Move-Item -LiteralPath $temporaryVhd -Destination $installedVhd

  Write-Host "[4/8] Importing the release VHDX..." -ForegroundColor Cyan
  Invoke-Checked "wsl.exe" @("--import-in-place", $DistributionName, $installedVhd)
  $script:createdDistribution = $true
  Invoke-Checked "wsl.exe" @("--manage", $DistributionName, "--set-default-user", "kenshin")

  Write-Host "[5/8] Running backend integrity and GPU preflight..." -ForegroundColor Cyan
  Invoke-Checked "wsl.exe" @(
    "-d", $DistributionName, "-u", "kenshin", "--cd", $BackendRoot, "--",
    $RuntimePython, "deployment/preflight.py", "--root", $BackendRoot,
    "--profile", "all", "--runtime-python", $RuntimePython, "--full-hash"
  )

  Write-Host "[6/8] Installing the Windows GUI and backend settings..." -ForegroundColor Cyan
  $guiSource = Join-Path $PayloadRoot $manifest.gui.file
  $guiDirectory = Join-Path $InstallRoot "gui"
  New-Item -ItemType Directory -Path $guiDirectory -Force | Out-Null
  $guiTarget = Join-Path $guiDirectory "Mask Pipeline Studio.exe"
  if (Test-Path -LiteralPath $guiTarget) {
    $script:guiBackup = Join-Path $InstallRoot "rollback\Mask Pipeline Studio.exe"
    New-Item -ItemType Directory -Path (Split-Path $script:guiBackup) -Force | Out-Null
    Copy-Item -LiteralPath $guiTarget -Destination $script:guiBackup -Force
  }
  Copy-Item -LiteralPath $guiSource -Destination $guiTarget -Force
  $settingsPath = Write-Settings $DistributionName

  Write-Host "[7/8] Running the Windows GUI to WSL end-to-end smoke test..." -ForegroundColor Cyan
  $fixtureSource = Join-Path $PayloadRoot $manifest.fixture.file
  $fixtureDirectory = Join-Path $InstallRoot "fixtures"
  New-Item -ItemType Directory -Path $fixtureDirectory -Force | Out-Null
  $fixtureTarget = Join-Path $fixtureDirectory "deployment-smoke.mp4"
  Copy-Item -LiteralPath $fixtureSource -Destination $fixtureTarget -Force
  $qaRoot = Join-Path $InstallRoot "validation"
  New-Item -ItemType Directory -Path $qaRoot -Force | Out-Null
  $qaReport = Join-Path $qaRoot "gui-e2e.json"
  $qaOutput = Join-Path $qaRoot "output"
  if (-not $SkipE2E) {
    Get-Process "Mask Pipeline Studio" -ErrorAction SilentlyContinue | Stop-Process -Force
    $arguments = @(
      "--software-rendering",
      "--qa-e2e-input=$fixtureTarget",
      "--qa-e2e-output=$qaOutput",
      "--qa-e2e-report=$qaReport",
      "--qa-e2e-max-frames=120"
    )
    $process = Start-Process -FilePath $guiTarget -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "GUI E2E failed with exit code $($process.ExitCode)" }
    # Windows PowerShell 5's ConvertFrom-Json can reject the intentionally
    # detailed (~1 MB) report when long log arrays contain mixed-width text.
    # The GUI process only exits 0 after its full internal assertions pass;
    # independently confirm that it also persisted the top-level pass state.
    $qaText = [IO.File]::ReadAllText($qaReport)
    if ($qaText -notmatch '(?m)^\s*"status"\s*:\s*"passed"\s*,?\s*$') {
      throw "GUI E2E report did not pass"
    }
  }

  Write-Host "[8/8] Creating shortcuts and deployment report..." -ForegroundColor Cyan
  # WSL distributions and GUI settings are per Windows user, so shortcuts
  # must remain per-user even when the launcher is started by an administrator.
  $desktop = [Environment]::GetFolderPath("DesktopDirectory")
  $programs = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
  New-Shortcut $guiTarget (Join-Path $desktop "Mask Pipeline Studio.lnk")
  New-Shortcut $guiTarget (Join-Path $programs "Mask Pipeline Studio.lnk")
  $report = [ordered]@{
    schema_version = 1
    status = "passed"
    installed_at_utc = [DateTime]::UtcNow.ToString("o")
    release_id = $manifest.release_id
    release_commit = $manifest.backend.release_commit
    asset_commit = $manifest.backend.asset_commit
    distribution = $DistributionName
    install_root = $InstallRoot
    backend_vhd = $installedVhd
    gui = $guiTarget
    gui_e2e = if ($SkipE2E) { "skipped" } else { $qaReport }
    gpu = $gpuName
    driver = $driverVersion
  }
  Write-Utf8Json (Join-Path $InstallRoot "deployment-report.json") $report 5
  Write-Host "Deployment passed: $InstallRoot" -ForegroundColor Green
  if (-not $NoLaunch) { Start-Process -FilePath $guiTarget | Out-Null }
} catch {
  # With ErrorActionPreference=Stop, Write-Error would itself terminate the
  # catch block and skip rollback. Report without raising, clean up first,
  # and rethrow the original failure after rollback has completed.
  Write-Host ("Deployment failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
  if (-not $KeepFailedInstall) {
    Write-Host "Rolling back the incomplete deployment..." -ForegroundColor Yellow
    if ($createdDistribution -and ((Get-Distros) -contains $DistributionName)) {
      & wsl.exe --terminate $DistributionName 2>$null | Out-Null
      & wsl.exe --unregister $DistributionName 2>$null | Out-Null
    }
    if ($settingsPath) { Restore-Settings $settingsPath }
    if ($guiBackup -and (Test-Path -LiteralPath $guiBackup)) {
      Copy-Item -LiteralPath $guiBackup -Destination (Join-Path $InstallRoot "gui\Mask Pipeline Studio.exe") -Force
    }
    if ($installedVhd -and (Test-Path -LiteralPath $installedVhd)) {
      Remove-Item -LiteralPath $installedVhd -Force
    }
  }
  throw
} finally {
  Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
