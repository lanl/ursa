#!/bin/sh
set -eu

upstream=${_URSA_HARBOR_UPSTREAM_BOOTSTRAP:-/staging/bootstrap-upstream.sh}
python=${_URSA_HARBOR_SYSTEM_PYTHON:-/usr/bin/python3}
if [ ! -x "$python" ]; then
  exec /bin/bash "$upstream" "$@"
fi
version=${_URSA_HARBOR_PYTHON_VERSION:-$("$python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')}

case "$version" in
  3.9)
    if ! "$python" -c 'import distutils.cmd' 2>/dev/null; then
      if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq python3-distutils 2>/dev/null || true
      elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache py3-setuptools 2>/dev/null || true
      elif command -v dnf >/dev/null 2>&1; then
        dnf install -y python3-setuptools 2>/dev/null || true
      elif command -v yum >/dev/null 2>&1; then
        yum install -y python3-setuptools 2>/dev/null || true
      fi
    fi
    patched=${_URSA_HARBOR_PATCHED_BOOTSTRAP:-/tmp/harbor-bootstrap.sh}
    sed 's|https://bootstrap.pypa.io/get-pip.py|https://bootstrap.pypa.io/pip/3.9/get-pip.py|g' \
      "$upstream" >"$patched"
    exec /bin/bash "$patched" "$@"
    ;;
  *) exec /bin/bash "$upstream" "$@" ;;
esac
