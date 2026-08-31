#!/usr/bin/env bash
# v11 = v9's optimization setup, started at QuestA's OWN hint ratios instead of bare.
#
# WHY: v9 and v10 both start every row at r=0 and let the Teacher raise doses where the
# model fails. Both plateaued around step 300-400 with ~1/3 of groups still yielding no
# gradient, and their probe curves are inside each other's noise. The minimal-dose start
# was our design choice, not QuestA's — this arm removes that difference.
#
# THE ONE CHANGE: MS_R0_BY_SRC. QuestA shipped two difficulty-filtered files, each row
# selected by scoring 0-4 out of 8 WITH a hint at that file's own prefix ratio:
#     OpenR1-50-0-4.jsonl   1,853 rows  ->  start at r=50   (14.8% of the pool)
#     OpenR1-25-0-4.jsonl  10,653 rows  ->  start at r=25   (85.2% of the pool)
# So every row begins at the dose its own filter certified as "hard but reachable", and
# nothing starts bare. The Teacher's job flips from raising doses to weaning them —
# which is the curriculum QuestA describes ("gradually reduce dependency on hints").
#
# CONSEQUENCE (deliberate): the high-dose budget is OFF. With 14.8% of the pool starting
# above r=25, a 10% budget is over-subscribed at step 0 and would clamp every promotion
# the Teacher ever proposes. That budget belongs to the minimal-dose design v11 is the
# alternative to; leaving it on would silently make this a different experiment.
#
# ALSO NOTE: nothing starts at r=0, so the mechanical graduation path (which only books
# problems already at dose 0) is inert until the Teacher weans something to 0. That is
# correct — graduation means "succeeds with no hint" — but it means early cycles have no
# mechanical moves at all, and every dose change is the Teacher's.
#
# Everything else is v9 verbatim: AdamW lr 2e-5 / betas (0.9,0.95) / wd 0.05 / eps 1e-5 as
# a set, advantage /std OFF, group size 8, DAPO dynamic filtering, no length penalty,
# 24K response cap, one optimizer update per step, original 12,506 rows undeduped.
# v10's true-Adam twin is not repeated here: at matched steps it tracks v9 inside the
# probe noise, so it is not the variable worth spending a run on.
#
# Launch:
#   cd /scratch/hongpaul-sandbox/mathscaffold && git pull
#   export MS_PYTHON=/scratch/hongpaul-sandbox/autokernel/kernel/bin/python3
#   export OPENAI_API_KEY=... WANDB_API_KEY=...
#   bash scripts/launch_b200_v11.sh teacher
set -eu
ARM=${1:-teacher}; CYCLES=${2:-80}

# ---- site (B200 pod) ---------------------------------------------------------------
export MS_ROOT=${MS_ROOT:-/scratch/hongpaul-sandbox/mathscaffold}
export MS_PYTHON=${MS_PYTHON:?export MS_PYTHON=/scratch/<you>/msenv/bin/python first}
export MS_MODEL=${MS_MODEL:-$MS_ROOT/models/OpenMath-Nemotron-1.5B}
export MS_CKPTS=${MS_CKPTS:-$MS_ROOT/ckpts}
export MS_DATA=${MS_DATA:-$MS_ROOT/data/questa_12k/OpenR1-25-0-4.jsonl,$MS_ROOT/data/questa_12k/OpenR1-50-0-4.jsonl}
export MS_N_GPUS=${MS_N_GPUS:-8}
export MS_WANDB=1 MS_WANDB_ENTITY=${MS_WANDB_ENTITY:-mhong-university-of-minnesota} MS_WANDB_PROJECT=${MS_WANDB_PROJECT:-mathscaffold}
export WANDB_ENTITY=$MS_WANDB_ENTITY WANDB_PROJECT=$MS_WANDB_PROJECT
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"
[ "$ARM" = teacher ] && : "${OPENAI_API_KEY:?teacher arm needs OPENAI_API_KEY}"

# ---- experiment (v11) ---------------------------------------------------------------
export MS_EXP=questa_${ARM}_v11 MS_WORK=$MS_ROOT/runs/${ARM}_v11 MS_WANDB_RUN_ID=questa_${ARM}_v11
export MS_LR=2e-5            # QuestA's lr — only safe together with eps below
export MS_BETA2=0.95 MS_WD=0.05 MS_EPS=1e-5   # their AReaL optimizer, transplanted as a SET
export MS_ADV_STD=False      # drop GRPO's /std (keep per-group centering)
export MS_N=8                # group size 8 (QuestA uses 16); halves tokens/step
export MS_FILTER_GROUPS=seq_final_reward MS_MAX_GEN_BATCHES=10
export MS_SERVE_MULT=2.5     # filtering eats ~1/accept_rate prompts per step; unconsumed rows
                             # return to the serving queue, so over-serving is free
# LENGTH PENALTY OFF. Empty MS_OVERLONG_LEN means train_stage.sh passes verl no overlong
# argument at all (its whole block is guarded by [ -n "$MS_OVERLONG_LEN" ]), so reward ==
# score with no length term. Faithful QuestA has no length penalty; here the eps-damped
# optimizer is the stabiliser under test, and a length term would confound it. To re-enable,
# set MS_OVERLONG_LEN=4096 (and optionally MS_OVERLONG_PENALTY, default 1.0) — deliberately
# NOT pre-set here, so nothing can silently switch it on.
export MS_OVERLONG_LEN=
export MS_MAXRESP=24000      # QuestA training cap (bare probe follows it; official probe stays 32K)
export MS_MINI_BS=128        # one optimizer update per step (= AReaL ppo_n_minibatches 1)
export MS_BS=128               # MS_N is set above (8); do not re-assign it here
export MS_NO_DEDUP=1
# v11 dosing: every row starts at the ratio ITS OWN QuestA file was filtered at (see the
# header). controller.load_state matches MS_R0_BY_SRC's substrings against each row's src,
# first rule wins; MS_R0 is only the fallback for a row matching neither, which with these
# two files never happens. Cap stays 50 (= the highest start), steps still <= 20.
export MS_R0_BY_SRC="50-0-4=50,25-0-4=25"
export MS_R0=25 MS_R_MAX=50 MS_MAX_DELTA=20 MS_PROMPT_STYLE=paper
# high-dose budget OFF (MS_HIGH_DOSE_R=0 disables the whole block in teacher.normalize):
# 14.8% of rows start above r=25, so a 10% budget would be full before the first decision.
export MS_HIGH_DOSE_R=0
export MS_BARE_SETS=$MS_ROOT/mathscaffold/bare_probe_sets_nodedup.json
export MS_SWITCH_CYCLE=${MS_SWITCH_CYCLE:-25}   # static arm: 50% -> 25% at cycle 25 (250 steps)
export MS_STALL_MIN=60       # stage watchdog: a 24K step incl. ckpt save is 15-30 min
# official AIME24/25 + HMMT25 probe every 2 cycles = every 20 steps (was 5 cycles = 50).
# The FSDP->HF merge is shared with the per-cycle bare probe (probe_ckpt.sh skips it when
# $CK/hf already exists), so the added cost is the eval itself: 90 problems x n=32 at 32K.
# The benchmark probe is now the ONLY probe. It runs each set twice — bare and at a 50%
# hint — so the hint-gain readout that the per-cycle bare probe used to provide comes from
# aime24 vs aime24_r50 and hmmt25 vs hmmt25_r50 instead, on the numbers we actually report.
# MS_BARE_PROBE=0 turns off the 200+200 training-distribution probe entirely; that also
# removes the held-out-minus-in-training gap, so memorization is no longer measured. The
# held-out rows stay excluded from serving anyway, which costs 2.3% of the pool and keeps
# the option of switching the bare probe back on mid-run without changing the training set.
export MS_PROBE_EVERY=2 MS_BARE_PROBE=0

# ---- preflight ---------------------------------------------------------------------
cd $MS_ROOT
for p in $MS_PYTHON $MS_MODEL/config.json ${MS_DATA%%,*} ${MS_DATA##*,} scripts/run_arm.sh; do
  [ -e "$p" ] || { echo "[preflight] missing: $p"; exit 3; }
done
if [ -e $MS_CKPTS/$MS_EXP ] || [ -e $MS_WORK ]; then
  # Resuming through this launcher rather than through launch.sh is deliberate: launch.sh
  # sets none of the arm's config (lr, eps, betas, /std, group size, filtering, response
  # cap, minibatch, dosing, high-dose budget), so a resume that bypasses this file silently
  # falls back to defaults and is a different experiment. MS_START_CYCLE=<C> keeps the whole
  # config and picks up from that cycle; the checkpoint and ratio state are already on disk.
  if [ -z "${MS_START_CYCLE:-}" ]; then
    echo "[preflight] $MS_CKPTS/$MS_EXP or $MS_WORK already exists: this launcher is FROM SCRATCH only."
    echo "            To resume with this arm's full config: MS_START_CYCLE=<next cycle> bash $0 $ARM"
    echo "            To start over: move both aside first."; exit 3
  fi
  echo "[preflight] resuming at cycle $MS_START_CYCLE (existing ckpt + ratio state kept)"
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
[ "$busy" = 0 ] || { echo "[preflight] $busy GPU(s) still hold memory:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3; }
$MS_PYTHON -c "import verl, vllm; print('[preflight] verl', verl.__version__, 'vllm', vllm.__version__)"
echo "[preflight] arm=$ARM exp=$MS_EXP work=$MS_WORK lr=$MS_LR beta2=${MS_BETA2:-0.999} wd=${MS_WD:-0.01} eps=${MS_EPS:-1e-8} advstd=${MS_ADV_STD:-True} maxresp=$MS_MAXRESP overlong=${MS_OVERLONG_LEN:-off} n=${MS_N:-16} filter=${MS_FILTER_GROUPS:-off} mini=$MS_MINI_BS R0=$MS_R0 by_src=$MS_R0_BY_SRC cap=$MS_R_MAX dmax=$MS_MAX_DELTA high=${MS_HIGH_DOSE_R}@${MS_HIGH_DOSE_FRAC} nodedup=$MS_NO_DEDUP cycles=$CYCLES git=$(git log --oneline -1 | cut -c1-40)"

# ---- launch ------------------------------------------------------------------------
bash scripts/launch.sh $ARM $CYCLES
echo "[v11] verify in ~3 min:  grep -E 'Training from scratch|actor/lr' $MS_WORK/logs/latest/train_c0.log | head -3"
echo "[v11] wandb runs: $MS_EXP (trainer), ${MS_EXP}_arm (cycle metrics), ${MS_EXP}_watch (liveness)"
