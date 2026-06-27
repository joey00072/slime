# Qwen3-1.7B DAPO 8k RL vs OPD on H200

This example reproduces the Slime DAPO systems comparison used in the OPD blog
on a 2xH200 node.

Run shape:

- Student: `Qwen/Qwen3-1.7B`
- Teacher for OPD: `Qwen/Qwen3-8B`
- Dataset: `zhuzilin/dapo-math-17k`
- Reward: DeepScaler rule reward
- OPD signal: SGLang teacher token logprobs
- Sequence shape: 8192 max response tokens
- Rollout batch: 32 prompts
- Group size: 8
- Samples per step: 256
- Steps: 30
- LoRA: disabled
- H200 layout: GPU 0 colocates Megatron actor training and student SGLang
  rollout; GPU 1 serves the Qwen3-8B teacher for OPD.

Run standard RL:

```bash
cd /root/slime
bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h200-deepscaler-opd.sh rl
```

Run OPD:

```bash
cd /root/slime
bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h200-deepscaler-opd.sh opd
```

The script downloads models and data, converts the student checkpoint to
Megatron `torch_dist` format if needed, starts the teacher server for OPD, and
submits the Slime Ray job.

Important environment overrides:

```bash
STUDENT_SGLANG_MEM_FRACTION=0.50
TEACHER_PORT=13141
NUM_ROLLOUT=30
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
ROLLOUT_MAX_RESPONSE_LEN=8192
MAX_TOKENS_PER_GPU=16384
```

The key custom piece is:

```text
slime.rollout.rm_hub.opd_deepscaler.post_process_rewards_with_deepscaler
```

Slime's default SGLang OPD post-process is pure distillation and returns zero
task rewards. This helper keeps the DeepScaler rule reward while attaching
teacher token logprobs to each sample, so OPD trains with both the math reward
and the online teacher signal.
