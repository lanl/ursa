#!/bin/bash
set -u
if test -d /opt/parthenon/.git \
  && git -C /opt/parthenon remote get-url origin | grep -Eiq 'lanl/parthenon(.git)?$' \
  && find /opt/parthenon/build -type f \( -name 'libparthenon.a' -o -name 'libparthenon.so' \) | grep -q .; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
