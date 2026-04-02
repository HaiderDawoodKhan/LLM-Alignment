# DVLM PA2 Alignment Pipeline

This repository implements a greenfield alignment stack for the PA2 assignment using:

- Policy/reference/aligned model: `HuggingFaceTB/SmolLM2-360M`
- Reward model backbone: `meta-llama/Llama-3.2-1B-Instruct`
- PPO value backbone: separate `meta-llama/Llama-3.2-1B-Instruct`
- LoRA only, no TRL, no `Trainer`, no ready-made PPO/DPO/GRPO trainers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure your Hugging Face token has access to the gated Llama model:

```bash
huggingface-cli login
```

## Training

Run the full assignment pipeline end to end:

```bash
bash run_all_tasks.sh
```

Optional skips:

```bash
bash run_all_tasks.sh --skip-c0 --skip-c7 --skip-c8
```

Reward model:

```bash
python3 train_rm.py
```

SFT warm-up:

```bash
python3 train_sft.py
```

RL / preference training:

```bash
python3 train_rl.py --method ppo
python3 train_rl.py --method dpo
python3 train_rl.py --method grpo
python3 train_rl.py --method rlvr
```

Evaluation:

```bash
python3 eval.py --models policy_sft ppo_policy dpo_policy grpo_policy rlvr_policy
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Outputs

- Logs: `logs/{run_name}/metrics.jsonl` and TensorBoard events
- Checkpoints: `checkpoints/policy_base`, `policy_sft`, `policy_ref`, `reward_model`, `ppo_policy`, `dpo_policy`, `grpo_policy`, `rlvr_policy`
