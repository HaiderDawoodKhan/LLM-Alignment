#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_C0=1
RUN_C7=1
RUN_C8=1

for arg in "$@"; do
  case "$arg" in
    --skip-c0)
      RUN_C0=0
      ;;
    --skip-c7)
      RUN_C7=0
      ;;
    --skip-c8)
      RUN_C8=0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--skip-c0] [--skip-c7] [--skip-c8]" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

run_step() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
  shift
  "$@"
}

if [[ "$RUN_C0" -eq 1 ]]; then
  run_step "Task C0: Data/model/tokenizer/LoRA verification" \
    "$PYTHON_BIN" - <<'PY'
from config import default_config
from model.policy import build_policy_tokenizer
from model.reward_model import build_reward_tokenizer
from seed import set_seed
from train_helpers import (
    build_c0_dataloaders,
    build_hh_datasets,
    data_verification,
    dataloader_verification,
    default_device,
    model_verification,
)

config = default_config(run_name="c0_verification")
set_seed(config.runtime.seed)
device = default_device(config)
datasets = build_hh_datasets(config)
data_verification(datasets, config.data.print_examples)
policy_tokenizer = build_policy_tokenizer(config.model)
reward_tokenizer = build_reward_tokenizer(config.model)
loaders = build_c0_dataloaders(datasets, policy_tokenizer, reward_tokenizer, config)
dataloader_verification(loaders)
model_verification(config, device)
PY
fi

run_step "Task C1: Reward model training" \
  "$PYTHON_BIN" train_rm.py

run_step "Task C2: SFT warm-up" \
  "$PYTHON_BIN" train_sft.py

run_step "Task C3: PPO training" \
  "$PYTHON_BIN" train_rl.py --method ppo

run_step "Task C4: DPO training" \
  "$PYTHON_BIN" train_rl.py --method dpo

run_step "Task C5: GRPO training" \
  "$PYTHON_BIN" train_rl.py --method grpo

run_step "Task C6: RLVR training" \
  "$PYTHON_BIN" train_rl.py --method rlvr

if [[ "$RUN_C7" -eq 1 ]]; then
  run_step "Task C7: GRPO KL beta sweep" \
    "$PYTHON_BIN" train_rl.py --method grpo --beta-sweep
fi

if [[ "$RUN_C8" -eq 1 ]]; then
  run_step "Task C8: Final evaluation" \
    "$PYTHON_BIN" eval.py --models ppo_policy dpo_policy grpo_policy rlvr_policy
fi

echo
echo "All requested tasks have finished."
