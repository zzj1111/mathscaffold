#!/usr/bin/env bash
# One training stage (to absolute STEP_TARGET, resume-aware).
# Machine-specific values come from env; every verl key below was smoke-proven on
# 4x H200 with pip-installed verl 0.7.0.dev (2026-08-14, runs/smoke_local3).
set -eu
STEP_TARGET=$1          # absolute target step
PARQUET=$2              # this cycle's train parquet
EXP=${MS_EXP:-questa_adaptive}
MODEL=${MS_MODEL:?set MS_MODEL to the OpenMath-Nemotron-1.5B path}
CKPTS=${MS_CKPTS:?set MS_CKPTS to the ckpt root}/$EXP
PY=${MS_PYTHON:-python3}

# ONE wandb run across all cycles: every per-cycle trainer process resumes the same
# run id (default MS_EXP) instead of minting a new one — the curve stays contiguous.
# Known cosmetic cost: a retry that re-walks already-logged steps has those points
# dropped by wandb's monotonic-step rule; the local logs keep the truth.
export WANDB_RUN_ID=${MS_WANDB_RUN_ID:-$EXP}
export WANDB_RESUME=${WANDB_RESUME:-allow}
# trainer run follows the SAME entity/project as the arm runs unless explicitly set —
# a relaunch shell that only exported MS_WANDB_* no longer strands the trainer run in
# the api key's default entity (seen live on B200: curve "vanished" into rl_agent)
export WANDB_ENTITY=${WANDB_ENTITY:-${MS_WANDB_ENTITY:-}}
export WANDB_PROJECT=${WANDB_PROJECT:-${MS_WANDB_PROJECT:-mathscaffold}}

# Optimizer matches QuestA's AReaL yaml (scripts/partial_50_grpo.yaml): AdamW lr 2e-5 constant,
# betas (0.9, 0.95), weight_decay 0.05, grad clip 1.0. verl's defaults were betas (0.9, 0.999) /
# wd 0.01 — beta2 0.999 reacts to a gradient-norm spike ~50x slower than 0.95, which is the
# leading suspect for the step-60-90 length collapse (grad_norm 0.02 -> 0.5-5 within a few steps).
# MS_PROMPT_STYLE=paper (default): the model's own chat template wraps the QuestA paper
# prompt (problem + ## Hint. prefix + boxed instruction) — see mathscaffold/data.py.
# MS_PROMPT_STYLE=repo_raw: the released add_prefix.py taken literally, fed as RAW TEXT via
# an identity chat template (contents only, no role markers/system/generation prompt);
# verl applies data.apply_chat_template_kwargs in both the dataset and the agent loop.
# Control only — it degenerates on OpenMath-Nemotron-1.5B (half the rollouts never stop).
RAW_TMPL='{% for m in messages %}{{ m.content }}{% endfor %}'
if [ "${MS_PROMPT_STYLE:-paper}" = "repo_raw" ]; then
  TMPL_ARGS=("++data.apply_chat_template_kwargs.chat_template='$RAW_TMPL'")
else
  TMPL_ARGS=()
fi

$PY -m verl.trainer.main_ppo "${TMPL_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$PARQUET \
    data.val_files=$PARQUET \
    data.train_batch_size=${MS_BS:-128} \
    data.max_prompt_length=${MS_MAXPROMPT:-8192} \
    data.max_response_length=${MS_MAXRESP:-32768} \
    data.filter_overlong_prompts=True \
    reward_model.enable=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=${MS_LR:-2e-5} \
    "actor_rollout_ref.actor.optim.betas=[0.9,${MS_BETA2:-0.95}]" \
    actor_rollout_ref.actor.optim.weight_decay=${MS_WD:-0.05} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MS_MINI_BS:-128} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MS_MICRO_BS:-2} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MS_LOGP_BS:-4} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MS_LOGP_BS:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${MS_N:-16} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${MS_GPU_UTIL:-0.7} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${MS_TP:-1} \
    custom_reward_function.path=${MS_ROOT:?set MS_ROOT}/mathscaffold/reward.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=${MS_N_GPUS:-8} trainer.nnodes=1 \
    trainer.save_freq=${MS_SAVE_FREQ:-10} trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=$STEP_TARGET \
    trainer.total_epochs=${MS_TOTAL_EPOCHS:-10000} \
    trainer.default_local_dir=$CKPTS \
    trainer.logger="${MS_TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name=${MS_WANDB_PROJECT:-mathscaffold} \
    trainer.experiment_name=$EXP \
    trainer.resume_mode=auto
