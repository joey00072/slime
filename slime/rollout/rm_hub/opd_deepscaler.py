import torch

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.utils.types import Sample


def post_process_rewards_with_deepscaler(args, samples: list[Sample], **kwargs):
    """Attach SGLang teacher logprobs while keeping DeepScaler task rewards.

    The default SGLang OPD post-process is pure distillation and returns zero
    scalar rewards. For DAPO-style math runs we want both signals:
    DeepScaler rule rewards for the task and teacher token logprobs for OPD.
    """
    raw_teacher_rewards = [sample.get_reward_value(args) for sample in samples]
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        for reward in raw_teacher_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-sample.response_length:]
        for t_log_prob, sample in zip(teacher_log_probs, samples, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    raw_rewards = [
        get_deepscaler_rule_based_reward(sample.response, sample.label)
        for sample in samples
    ]
    rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    if args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        rewards = rewards - rewards.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
        rewards = rewards.flatten()

    return raw_rewards, rewards.tolist()
