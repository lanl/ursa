#!/bin/bash
set -u
if python3 -c 'import gpaw; assert gpaw.__version__'; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
