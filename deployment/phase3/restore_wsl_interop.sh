#!/usr/bin/env bash
set -euo pipefail

registration=/proc/sys/fs/binfmt_misc/WSLInterop
if [[ -e "$registration" ]]; then
  exit 0
fi
if [[ $(id -u) -ne 0 ]]; then
  echo "restore_wsl_interop.sh must run as root" >&2
  exit 1
fi
printf '%s' ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register
[[ -e "$registration" ]] || {
  echo "failed to restore WSLInterop binfmt registration" >&2
  exit 1
}
