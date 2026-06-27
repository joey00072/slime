#!/usr/bin/env bash
set -euo pipefail

# Reproduce the DAPO 8k RL-vs-OPD systems experiment on a 2xH200 node.
#
# Default layout:
#   GPU 0: Megatron actor training colocated with student SGLang rollout.
#   GPU 1: Qwen3-8B SGLang teacher for OPD.
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h200-deepscaler-opd.sh rl
#   bash examples/on_policy_distillation/run-qwen3-1.7B-dapo-h200-deepscaler-opd.sh opd

MODE="${1:-rl}"
if [[ "$MODE" != "rl" && "$MODE" != "opd" ]]; then
  echo "usage: $0 rl|opd" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export TEACHER_PORT="${TEACHER_PORT:-13141}"

export STUDENT_MODEL_DIR="${STUDENT_MODEL_DIR:-/root/models/Qwen3-1.7B}"
export TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-/root/models/Qwen3-8B}"
export STUDENT_DIST_DIR="${STUDENT_DIST_DIR:-/root/models/Qwen3-1.7B_torch_dist}"
export DATA_DIR="${DATA_DIR:-/root/datasets/dapo-math-17k}"
export RUN_ROOT="${RUN_ROOT:-/root/slime-runs/qwen3-1.7b-dapo-h200-${MODE}}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-30}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export STUDENT_SGLANG_MEM_FRACTION="${STUDENT_SGLANG_MEM_FRACTION:-0.50}"
export TEACHER_SGLANG_MEM_FRACTION="${TEACHER_SGLANG_MEM_FRACTION:-0.60}"

mkdir -p /root/models /root/datasets "$RUN_ROOT"
cd /root/slime

pip install -e . --no-deps >/tmp/slime_editable_install.log 2>&1 || {
  cat /tmp/slime_editable_install.log
  exit 1
}

cleanup() {
  if [[ -n "${TEACHER_PID:-}" ]]; then
    kill "$TEACHER_PID" 2>/dev/null || true
  fi
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  pkill -9 slime 2>/dev/null || true
  pkill -9 redis 2>/dev/null || true
}
trap cleanup EXIT
cleanup
sleep 3

echo "Preparing models and DAPO data"
hf download Qwen/Qwen3-1.7B --local-dir "$STUDENT_MODEL_DIR"
if [[ "$MODE" == "opd" ]]; then
  hf download Qwen/Qwen3-8B --local-dir "$TEACHER_MODEL_DIR"
fi
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "$DATA_DIR"

if [[ ! -f "$STUDENT_DIST_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Converting student HF checkpoint to Megatron torch_dist"
  # shellcheck source=/root/slime/scripts/models/qwen3-1.7B.sh
  source /root/slime/scripts/models/qwen3-1.7B.sh
  PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 1 /root/slime/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$STUDENT_MODEL_DIR" \
    --save "$STUDENT_DIST_DIR"
else
  echo "Using existing converted checkpoint: $STUDENT_DIST_DIR"
fi

TEACHER_PID=""
if [[ "$MODE" == "opd" ]]; then
  TEACHER_LOG="$RUN_ROOT/teacher_sglang.log"
  echo "Starting Qwen3-8B OPD teacher on GPU 1"
  CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$TEACHER_PORT" \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "$TEACHER_SGLANG_MEM_FRACTION" \
    > "$TEACHER_LOG" 2>&1 &
  TEACHER_PID="$!"

  TEACHER_READY=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
      echo "Teacher exited early; tail follows" >&2
      tail -120 "$TEACHER_LOG" >&2 || true
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
      curl -sf "http://127.0.0.1:${TEACHER_PORT}/get_model_info" || true
      TEACHER_READY=1
      break
    fi
    tail -20 "$TEACHER_LOG" || true
    sleep 5
  done
  if [[ "$TEACHER_READY" != "1" ]]; then
    echo "Teacher did not become healthy; tail follows" >&2
    tail -120 "$TEACHER_LOG" >&2 || true
    exit 1
  fi
fi

# shellcheck source=/root/slime/scripts/models/qwen3-1.7B.sh
source /root/slime/scripts/models/qwen3-1.7B.sh

CKPT_ARGS=(
  --hf-checkpoint "$STUDENT_MODEL_DIR"
  --ref-load "$STUDENT_DIST_DIR"
  --load "$RUN_ROOT/checkpoints"
  --save "$RUN_ROOT/checkpoints"
  --save-interval 1000
)

ROLLOUT_ARGS=(
  --prompt-data "$DATA_DIR/dapo-math-17k.jsonl"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --num-steps-per-rollout 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-temperature 1
  --balance-data
)

RM_ARGS=(--rm-type deepscaler)
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

if [[ "$MODE" == "opd" ]]; then
  RM_ARGS=(
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path slime.rollout.rm_hub.opd_deepscaler.post_process_rewards_with_deepscaler
    --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"
  )
  GRPO_ARGS+=(
    --use-opd
    --opd-type sglang
    --opd-kl-coef 1.0
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "$STUDENT_SGLANG_MEM_FRACTION"
  --sglang-cuda-graph-max-bs 32
  --sglang-enable-metrics
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  --num-gpus-per-node 1
  --colocate
)

export CUDA_VISIBLE_DEVICES=0
ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus 1 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"RAY_USE_UVLOOP\": \"0\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 /root/slime/train.py \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}" 2>&1 | tee "$RUN_ROOT/ray_job.log"
