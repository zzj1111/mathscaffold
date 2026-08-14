#!/usr/bin/env bash
# One training stage (K steps, resume-aware) on 8xB200. Machine-specific values
# marked TODO — fill in on the kernel machine (its verl checkout + conda env).
set -eu
STEP_TARGET=$1          # absolute target step
PARQUET=$2              # this cycle's train parquet
EXP=${MS_EXP:-questa_adaptive}
MODEL=${MS_MODEL:-/path/to/OpenMath-Nemotron-1.5B}          # TODO
CKPTS=${MS_CKPTS:-/path/to/ckpts}/$EXP                      # TODO
VERL=${MS_VERL:-/path/to/verl}                              # TODO

cd $VERL
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$PARQUET \
    data.val_files=${MS_VAL_PARQUET:-$PARQUET} \
    data.train_batch_size=${MS_BS:-128} \
    data.max_prompt_length=4096 \
    data.max_response_length=24000 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.rollout.n=${MS_N:-16} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    custom_reward_function.path=$MS_ROOT/mathscaffold/reward.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 \
    trainer.save_freq=${MS_SAVE_FREQ:-8} \
    trainer.default_local_dir=$CKPTS \
    trainer.total_training_steps=$STEP_TARGET \
    trainer.resume_mode=auto \
    trainer.project_name=mathscaffold trainer.experiment_name=$EXP
