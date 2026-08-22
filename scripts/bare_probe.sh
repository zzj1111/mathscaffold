#!/usr/bin/env bash
# Per-cycle hint-free (bare) probe on two fixed 200-problem sets from the TRAINING
# distribution (mathscaffold/bare_probe_sets.json): `heldout` (never trained on) and
# `train` (in the rotation). Prompt = the training prompt at ratio 0, sampling = the
# training distribution (temperature 1.0, max response = training cap), n=MS_BARE_N
# samples per problem, scored by the training reward. ~5-8 min on 8 GPUs.
#
# Appends one record per cycle to $MS_WORK/bare_probe.jsonl and rewrites
# $MS_WORK/bare_probe.json (latest). The teacher preamble (prepare_cycle.py) and
# wandb (wb.py) read them: hinted success rising while these stall = the policy is
# learning to continue hints, not to solve. Called by run_arm.sh after every
# MS_BARE_EVERY cycles (default 1); MS_BARE_PROBE=0 disables. Safe to run by hand:
#   MS_ROOT=... MS_CKPTS=... MS_EXP=... MS_WORK=... bash scripts/bare_probe.sh <cycle>
set -eu
CYCLE=$1
ROOT=${MS_ROOT:?}; WORK=${MS_WORK:-$ROOT/runs/${MS_ARM:-teacher}}; PY=${MS_PYTHON:-python3}
EXP=${MS_EXP:-questa_teacher}; CKPTS=${MS_CKPTS:?}/$EXP
PORT=${MS_BARE_PORT:-8143}
N=${MS_BARE_N:-4}
STEP=$(cat $CKPTS/latest_checkpointed_iteration.txt)
CK=$CKPTS/global_step_$STEP
HF=$CK/hf
if [ ! -f $HF/config.json ]; then
  $PY -m verl.model_merger merge --backend fsdp --local_dir $CK/actor --target_dir $HF \
    || $PY ${MS_VERL_ROOT:-.}/scripts/model_merger.py merge --backend fsdp --local_dir $CK/actor --target_dir $HF
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json generation_config.json added_tokens.json chat_template.jinja; do
    [ -f $MS_MODEL/$f ] && [ ! -f $HF/$f ] && cp $MS_MODEL/$f $HF/ || true
  done
fi

LOGDIR=${LOGDIR:-$WORK/logs/latest}; mkdir -p $LOGDIR/bare
source $ROOT/scripts/vllm_pool.sh
# server context must hold prompt (<=8192) + the training response cap
LOGPREFIX=$LOGDIR/bare/vllm_c${CYCLE} MAXLEN=${MS_BARE_MAXLEN:-$(( ${MS_MAXRESP:-32768} + 8192 ))} UTIL=0.85
ms_pool_start || { echo "[bare] FAIL: GPUs not free"; exit 3; }
trap ms_pool_stop EXIT
ms_pool_wait 1500 || { echo "[bare] FAIL: vLLM servers did not come up"; exit 3; }
echo "[bare] cycle $CYCLE step $STEP: n=$N servers $URLS"

SETS=$ROOT/mathscaffold/bare_probe_sets.json
OUT=$LOGDIR/bare/c${CYCLE}
for which in heldout train; do
  $PY - "$SETS" "$which" > $OUT.$which.qids.json <<'EOF'
import json, sys; print(json.dumps(json.load(open(sys.argv[1]))[sys.argv[2]]))
EOF
  $PY $ROOT/scripts/eval_bare_trainset.py --model-dir $HF --base-url "$URLS" \
      --jsonl "${MS_DATA:?}" --qids $OUT.$which.qids.json --n $N --ratio 0 \
      --max-tokens ${MS_MAXRESP:-32768} --out $OUT.$which 2>&1 | tail -n 3
done

$PY - "$WORK" "$CYCLE" "$STEP" "$N" "$OUT" <<'EOF'
import json, sys, time
work, cycle, step, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rec = {"cycle": cycle, "step": step, "n": n, "time": time.strftime("%Y-%m-%d %H:%M")}
for which in ("heldout", "train"):
    s = json.load(open(f"{out}.{which}.summary.json"))
    rec[which] = {"pass1": s["mean_pass1"], "stderr": s["stderr"], "solved_any": s["solved_any"],
                  "truncated": s["truncated_frac"], "mean_chars": s["mean_chars"], "n_problems": s["n_problems"]}
rec["gap_train_minus_heldout"] = round(rec["train"]["pass1"] - rec["heldout"]["pass1"], 4)
with open(f"{work}/bare_probe.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
json.dump(rec, open(f"{work}/bare_probe.json", "w"))
print(f"[bare] cycle {cycle} step {step}: heldout {rec['heldout']['pass1']:.3f} (±{rec['heldout']['stderr']:.3f})  "
      f"train {rec['train']['pass1']:.3f} (±{rec['train']['stderr']:.3f})  gap {rec['gap_train_minus_heldout']:+.3f}  "
      f"trunc {rec['heldout']['truncated']:.1%}/{rec['train']['truncated']:.1%}")
EOF
