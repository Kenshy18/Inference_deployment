#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
powershell_script=$(wslpath -w "$script_dir/../windows/Start-WindowsDev.ps1")

exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$powershell_script" "$@"
