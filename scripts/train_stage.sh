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

# Restart with a NEW optimizer (e.g. a changed lr): seed $CKPTS/global_step_<N>/actor with the
# model shards only (no optim_*/extra_state_* files) and latest_checkpointed_iteration.txt=N.
# verl then loads the weights, keeps global step N (folder name) and builds a fresh optimizer
# + lr scheduler from this config. Loading the saved optimizer/extra instead would silently
# restore the OLD lr (Optimizer/LambdaLR.load_state_dict carry param_groups/base_lrs).
# Auto-detected per stage, so every later stage (whose ckpts have optimizer shards) resumes fully.
LOAD_ARGS=()
LATEST=$(cat $CKPTS/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)

# verl restores the StatefulDataLoader position from the checkpoint's data.pt. Every cycle
# trains on a DIFFERENT parquet, so that position is meaningless here: measured live, a new
# cycle resumes at row <previous consumption> and silently SKIPS that many rows of its own
# slice — i.e. most of the problems the teacher just dosed never get trained. It only looked
# harmless while consumption exactly equalled the slice size (10 steps x 128 = 1280 rows);
# MS_SERVE_MULT and dynamic filtering both break that equality.
# Rule: drop data.pt iff the parquet changed since the last invocation. A retry of the SAME
# cycle keeps it (correctly resuming mid-slice); the first attempt of a new cycle drops the
# stale one, so retries can never see a stale state afterwards.
LAST_RUN=$CKPTS/last_train_parquet.txt
if [ "$LATEST" -gt 0 ] && [ "$(cat $LAST_RUN 2>/dev/null)" != "$PARQUET" ]; then
  if [ -f $CKPTS/global_step_$LATEST/data.pt ]; then
    mv $CKPTS/global_step_$LATEST/data.pt $CKPTS/global_step_$LATEST/data.pt.prev_parquet
    echo "[stage] new parquet -> dropped the checkpoint's dataloader state (it points into the previous slice)"
  fi
fi
mkdir -p $CKPTS && echo "$PARQUET" > $LAST_RUN
if [ "$LATEST" -gt 0 ] && ! ls $CKPTS/global_step_$LATEST/actor/optim_world_size_* >/dev/null 2>&1; then
  LOAD_ARGS=("actor_rollout_ref.actor.checkpoint.load_contents=['model']")
  echo "[stage] ckpt global_step_$LATEST has no optimizer shards: loading model only -> fresh optimizer at lr ${MS_LR:-2e-5}"
fi

# Optional soft length penalty (DAPO overlong buffer, verl reward manager "dapo"): the reward
# of a response longer than MS_MAXRESP - MS_OVERLONG_LEN decreases linearly, reaching
# -MS_OVERLONG_PENALTY at the cap; shorter responses are untouched. Gives the policy a
# "wrap up" gradient BEFORE truncation instead of the reward-0 cliff at the cap (the cliff
# fed the length runaway). Off unless MS_OVERLONG_LEN is set. The rollout log keeps the raw
# correctness (reward.py writes its own score before the manager adds the penalty), so the
# teacher's pass/fail bookkeeping is unaffected; wandb gets acc (raw) and overlong_reward.
OVERLONG_ARGS=()
if [ -n "${MS_OVERLONG_LEN:-}" ]; then
  OVERLONG_ARGS=(reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$MS_OVERLONG_LEN
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${MS_OVERLONG_PENALTY:-1.0}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True
    +reward_model.reward_kwargs.max_resp_len=${MS_MAXRESP:-32768})
  echo "[stage] overlong penalty on: linear from $(( ${MS_MAXRESP:-32768} - MS_OVERLONG_LEN )) to ${MS_MAXRESP:-32768} tokens, -${MS_OVERLONG_PENALTY:-1.0} at the cap"
fi

# DAPO-style dynamic filtering (QuestA/AReaL do this): a prompt group whose rollouts are
# all-correct or all-wrong carries no within-group contrast, so it is dropped and the
# trainer keeps pulling prompts until it has train_batch_size groups with mixed rewards.
# Only recipe/dapo's trainer implements it (main_ppo does not), and that trainer always
# uses the agent-loop rollout manager — which is the mode we already run in. Its config
# inherits ppo_trainer, so every override below is unchanged.
# MS_FILTER_GROUPS=<metric> turns it on (metric: seq_final_reward | seq_reward | acc).
# NOTE: with filtering a cycle eats ~steps*batch/accept_rate prompts, so serve a bigger
# slice (MS_SERVE_MULT in prepare_cycle.py); unconsumed rows return to the serving queue.
ENTRY=verl.trainer.main_ppo
FILTER_ARGS=()
if [ -n "${MS_FILTER_GROUPS:-}" ]; then
  ENTRY=recipe.dapo.main_dapo
  FILTER_ARGS=(algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=$MS_FILTER_GROUPS
    algorithm.filter_groups.max_num_gen_batches=${MS_MAX_GEN_BATCHES:-10}
    data.gen_batch_size=${MS_GEN_BS:-${MS_BS:-128}})
  VERL_PKG=$($PY -c "import verl,os;print(os.path.dirname(verl.__file__))")
  # recipe/ lives next to the verl package; and dapo_trainer.yaml inherits ppo_trainer via a
  # RELATIVE hydra searchpath ("file://verl/trainer/config"), which only resolves when the cwd
  # is the verl checkout — we launch from MS_ROOT, so point it at the absolute path instead.
  export PYTHONPATH=${MS_VERL_ROOT:-$(dirname $VERL_PKG)}:${PYTHONPATH:-}
  FILTER_ARGS+=("hydra.searchpath=[file://$VERL_PKG/trainer/config]")
  echo "[stage] dynamic filtering ON (metric=$MS_FILTER_GROUPS, max_gen_batches=${MS_MAX_GEN_BATCHES:-10}) via $ENTRY"
fi

$PY -m $ENTRY "${TMPL_ARGS[@]}" "${LOAD_ARGS[@]}" "${OVERLONG_ARGS[@]}" "${FILTER_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    ${MS_ADV_STD:+algorithm.norm_adv_by_std_in_grpo=$MS_ADV_STD} \
    data.train_files=$PARQUET \
    data.val_files=$PARQUET \
    data.train_batch_size=${MS_BS:-128} \
    data.max_prompt_length=${MS_MAXPROMPT:-8192} \
    data.max_response_length=${MS_MAXRESP:-32768} \
    data.filter_overlong_prompts=True \
    reward_model.enable=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=${MS_LR:-2e-5} \
    ${MS_BETA2:+"actor_rollout_ref.actor.optim.betas=[0.9,$MS_BETA2]"} \
    ${MS_WD:+actor_rollout_ref.actor.optim.weight_decay=$MS_WD} \
    ${MS_EPS:++actor_rollout_ref.actor.optim.override_optimizer_config.eps=$MS_EPS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MS_MINI_BS:-128} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MS_MICRO_BS:-2} \
    actor_rollout_ref.actor.use_kl_loss=$([ -n "${MS_KL:-}" ] && echo True || echo False) \
    ${MS_KL:+actor_rollout_ref.actor.kl_loss_coef=$MS_KL} \
    ${MS_CLIP_LOW:+actor_rollout_ref.actor.clip_ratio_low=$MS_CLIP_LOW} \
    ${MS_CLIP_HIGH:+actor_rollout_ref.actor.clip_ratio_high=$MS_CLIP_HIGH} \
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
