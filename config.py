from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    hh_dataset_name: str = "Anthropic/hh-rlhf"
    hh_dataset_config: Optional[str] = "harmless-base"
    gsm8k_dataset_name: str = "openai/gsm8k"
    gsm8k_dataset_config: str = "main"
    max_seq_len: int = 1024
    max_prompt_tokens: int = 896
    max_new_tokens: int = 128
    rlvr_max_new_tokens: int = 256
    train_split: str = "train"
    eval_split: str = "test"
    eval_subset_size: int = 200
    print_examples: int = 3


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelConfig:
    policy_name: str = "HuggingFaceTB/SmolLM2-360M"
    rm_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    value_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    policy_padding_side: str = "left"
    rm_padding_side: str = "right"
    trust_remote_code: bool = False
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    quantize_frozen_models: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class RMConfig:
    epochs: int = 1
    batch_size: int = 8
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    lambda_reg: float = 1e-3
    eval_every: int = 50


@dataclass
class SFTConfig:
    epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 4
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=2e-4))
    eval_every: int = 100


@dataclass
class PPOConfig:
    num_updates: int = 200
    prompts_per_step: int = 8
    minibatch_epochs: int = 4
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    value_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=5e-5))
    gamma: float = 1.0
    gae_lambda: float = 0.95
    beta_kl: float = 0.1
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    eval_every: int = 25
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class DPOConfig:
    epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 4
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    beta: float = 0.1
    eval_every: int = 25


@dataclass
class GRPOConfig:
    num_updates: int = 200
    prompts_per_step: int = 8
    group_size: int = 4
    minibatch_epochs: int = 4
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    beta_kl: float = 0.1
    clip_epsilon: float = 0.2
    eval_every: int = 25
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class RLVRConfig:
    num_updates: int = 300
    prompts_per_step: int = 8
    group_size: int = 4
    minibatch_epochs: int = 4
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=5e-6))
    beta_kl: float = 0.05
    clip_epsilon: float = 0.2
    eval_every: int = 25
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class EvalConfig:
    num_prompts: int = 200
    sample_prompts: int = 5
    greedy_temperature: float = 0.0
    resource_log_window: int = 20


@dataclass
class AblationConfig:
    ppo_grpo_beta: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.5])
    ppo_grpo_clip: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.5, float("inf")])
    grpo_group_size: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dpo_beta: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1.0])


@dataclass
class RuntimeConfig:
    seed: int = 42
    output_dir: str = "checkpoints"
    logs_dir: str = "logs"
    run_name: str = "default"
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rlvr: RLVRConfig = field(default_factory=RLVRConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def checkpoint_dir(self, name: str) -> Path:
        return Path(self.runtime.output_dir) / name

    def log_dir(self, run_name: Optional[str] = None) -> Path:
        target = run_name or self.runtime.run_name
        return Path(self.runtime.logs_dir) / target

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_config(run_name: str = "default") -> AppConfig:
    cfg = AppConfig()
    cfg.runtime.run_name = run_name
    return cfg
