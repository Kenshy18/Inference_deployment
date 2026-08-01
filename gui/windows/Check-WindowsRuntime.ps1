param(
  [string]$Distribution = "Ubuntu-24.04",
  [string]$BackendRoot = "/home/kenshin/inference_backend2",
  [string]$RuntimePython = "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10"
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name, [string]$InstallHint) {
  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $command) {
    throw "$Name was not found. $InstallHint"
  }
  return $command.Source
}

$wsl = Require-Command "wsl.exe" "Enable WSL2 first."
$node = Require-Command "node.exe" "Install Node.js 20 or later for Windows."
$npm = Require-Command "npm.cmd" "Install npm together with Windows Node.js."

$nodeVersion = (& $node --version).Trim().TrimStart("v")
$nodeMajor = [int]($nodeVersion.Split(".")[0])
if ($nodeMajor -lt 20) {
  throw "Windows Node.js 20 or later is required; found $nodeVersion."
}

$distributions = @(& $wsl --list --quiet) | ForEach-Object { $_.Trim("`0 ") }
if ($distributions -notcontains $Distribution) {
  throw "WSL distribution '$Distribution' was not found."
}

& $wsl -d $Distribution -- /usr/bin/test -d $BackendRoot -a -x $RuntimePython
if ($LASTEXITCODE -ne 0) {
  throw "The WSL backend root or runtime Python was not found."
}

& $wsl -d $Distribution --cd $BackendRoot -- $RuntimePython -c "import orchestration; print('orchestration import: OK')"
if ($LASTEXITCODE -ne 0) {
  throw "The WSL orchestration import check failed."
}

[pscustomobject]@{
  WindowsNode = $nodeVersion
  NodePath = $node
  NpmPath = $npm
  WslPath = $wsl
  Distribution = $Distribution
  BackendRoot = $BackendRoot
  RuntimePython = $RuntimePython
} | Format-List

Write-Host "Windows build runtime is ready." -ForegroundColor Green
