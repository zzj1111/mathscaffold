#!/usr/bin/env bash
# Local (4x H200) end-to-end smoke of the math arm: prepare -> 2 GRPO steps with
# vLLM rollouts + Math-Verify reward + recorder -> controller consumes real
# outcomes. Small on purpose: 64 problems, n=8, 4k generation.
set -eu
ROOT=${MS_ROOT:-/home/zha00175/mathscaffold}
WORK=${MS_WORK:-$ROOT/runs/smoke_local3}
MODEL=${MS_MODEL:-/mnt/data1/zha00175/math_prep/OpenMath-Nemotron-1.5B}
CKPTS=${MS_CKPTS:-/mnt/data1/zha00175/rebuild_runs/ckpts/math_smoke_local}
PY=${MS_PYTHON:-/home/zha00175/venv_verl/bin/python}
mkdir -p $WORK
cd $ROOT

export MS_K=2 MS_BS=32
$PY scripts/prepare_cycle.py --arm adaptive --cycle 0 \
    --state $WORK/ratio_state.json --out $WORK/train_c0.parquet 2>&1 | tee $WORK/prep0.log

export MATHSCAFFOLD_ROLLOUT_LOG=$WORK/rollouts_c0.jsonl
export CUDA_VISIBLE_DEVICES=${MS_GPUS:-0,1,6,7}
$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$WORK/train_c0.parquet \
    data.val_files=$WORK/train_c0.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=${MS_MAXRESP:-16384} \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    reward_model.enable=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$ROOT/mathscaffold/reward.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=${MS_N_GPUS:-4} trainer.nnodes=1 \
    trainer.save_freq=2 trainer.test_freq=-1 trainer.val_before_train=False \
    trainer.total_training_steps=2 \
    trainer.default_local_dir=$CKPTS \
    trainer.logger='["console"]' \
    trainer.project_name=mathscaffold trainer.experiment_name=math_smoke_local \
    trainer.resume_mode=disable 2>&1 | tee $WORK/train_c0.log

# 控制器消费真实结果
$PY scripts/prepare_cycle.py --arm adaptive --cycle 1 \
    --state $WORK/ratio_state.json --rollout-log $WORK/rollouts_c0.jsonl \
    --out $WORK/train_c1.parquet 2>&1 | tee $WORK/prep1.log

$PY - <<'PYEOF'
import json, collections
WORK = "/home/zha00175/mathscaffold/runs/smoke_local3"
rows = [json.loads(x) for x in open(f"{WORK}/rollouts_c0.jsonl")]
byq = collections.defaultdict(list)
for r in rows: byq[r["qid"]].append(r["score"])
st = json.load(open(f"{WORK}/ratio_state.json"))
moved = sum(1 for q, h in st["problems"].items()
            if q in byq and h["r"] != 50.0)
print(f"[SMOKE] recorder {len(rows)} 行 / {len(byq)} 题; "
      f"成功率 {sum(r['score'] for r in rows)/len(rows):.3f}; "
      f"控制器已调整 {moved} 题的 r")
assert len(rows) >= 64 * 8 * 0.9, "recorder 行数不足"
print("[SMOKE] PASS")
PYEOF
echo "[SMOKE] all done"
