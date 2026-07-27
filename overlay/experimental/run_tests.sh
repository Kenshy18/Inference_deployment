#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/build.sh" >/dev/null
python3 -m unittest discover -s "$script_dir/tests" -v
