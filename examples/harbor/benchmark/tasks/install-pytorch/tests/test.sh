#!/bin/bash
set -u
if python3 -c 'import torch; assert torch.tensor([1, 2]).sum().item() == 3'; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
