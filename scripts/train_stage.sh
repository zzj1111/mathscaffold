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

$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$PARQUET \
    data.val_files=$PARQUET \
    data.train_batch_size=${MS_BS:-128} \
    data.max_prompt_length=${MS_MAXPROMPT:-8192} \
    data.max_response_length=${MS_MAXRESP:-24000} \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    reward_model.enable=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=${MS_LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MS_MINI_BS:-32} \
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
    trainer.default_local_dir=$CKPTS \
    trainer.logger="${MS_TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name=${MS_WANDB_PROJECT:-mathscaffold} \
    trainer.experiment_name=$EXP \
    trainer.resume_mode=auto
