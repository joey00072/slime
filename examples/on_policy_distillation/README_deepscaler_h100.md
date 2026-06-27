# Qwen3-1.7B DAPO 8k RL vs OPD on H100

This example reproduces the DAPO systems comparison used in our OPD blog work,
but with an H100-friendly GPU layout.

The H200 run colocated Megatron training and student SGLang rollout on one
143GB H200, then placed the Qwen3-8B teacher on a second H200. That exact layout
is too tight for normal 80GB H100 nodes. This H100 script keeps the same
experiment shape but splits the work across more GPUs.

Default run shape:

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
- H100 layout: 2 training GPUs, 4 student rollout GPUs, 2 teacher GPUs

Run standard RL:

```bash
cd /root/slime
bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h100-deepscaler-opd.sh rl
```

Run OPD:

```bash
cd /root/slime
bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h100-deepscaler-opd.sh opd
```

The script downloads models and data, converts the student checkpoint to
Megatron `torch_dist` format if needed, starts the teacher server for OPD, and
submits the Slime Ray job.

Important environment overrides:

```bash
RAY_VISIBLE_DEVICES=0,1,2,3,4,5
TEACHER_VISIBLE_DEVICES=6,7
ACTOR_GPUS=2
ROLLOUT_GPUS=4
TEACHER_TP=2
NUM_ROLLOUT=30
GLOBAL_BATCH_SIZE=256
ROLLOUT_MAX_RESPONSE_LEN=8192
```

For a smaller H100 node, lower the batch size, response length, or rollout
parallelism deliberately. The default is intended to preserve the blog
experiment's 8k, group-8, batch-256 shape.

The key custom piece is:

```text
slime.rollout.rm_hub.opd_deepscaler.post_process_rewards_with_deepscaler
```

Slime's default SGLang OPD post-process is pure distillation and returns zero
task rewards. This helper keeps the DeepScaler rule reward while attaching
teacher token logprobs to each sample, so OPD trains with both the math reward
and the online teacher signal.
