#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../.." && pwd)
powershell_script=$(wslpath -w "$script_dir/../windows/Build-Windows.ps1")

exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$powershell_script" \
  -RepositoryRoot "$repository_root" "$@"
