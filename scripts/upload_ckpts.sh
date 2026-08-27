#!/usr/bin/env bash
# 一次性把 v5/v6/v7 的 ckpt 传到 HF。可与 v9/v10 训练并行运行:
#   * CUDA_VISIBLE_DEVICES= 强制合并走 CPU,绝不抢训练的卡
#   * nice/ionice 降优先级,不和 trainer 抢 CPU/IO
#   * 合并到临时目录 -> 上传 -> 立刻删除,磁盘峰值只占一个 ckpt
#   * 已传过的自动跳过,断了重跑即可续
#
# 用法(在 B200 上):
#   HF_TOKEN=hf_xxx bash upload_ckpts.sh                  # 权重(每 ckpt ~3GB)
#   HF_TOKEN=hf_xxx WITH_OPTIM=1 bash upload_ckpts.sh     # 额外传 v7 崩溃点的优化器状态(每个 ~18GB)
#   HF_TOKEN=hf_xxx ALL_STEPS=1 bash upload_ckpts.sh      # 不做步数抽样,全传
set -u
REPO=${REPO:-Mingyi-Hong/mathscaffold-ckpts}
: "${HF_TOKEN:?export HF_TOKEN=hf_... 先设 token}"
: "${MS_CKPTS:?MS_CKPTS 未设 — 先 source 你的 launch 配置}"
PY=${MS_PYTHON:-python}
TMP=${TMP_MERGE:-$MS_CKPTS/_upload_tmp}
LOG=${LOG:-$HOME/upload_ckpts_$(date +%m%d_%H%M).log}
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%H:%M)] repo=$REPO  ckpts=$MS_CKPTS  log=$LOG"

# 已在 HF 上的路径,用于跳过
HAVE=$($PY - <<'PY' 2>/dev/null
import os
from huggingface_hub import HfApi
try:
    fs = HfApi().list_repo_files(os.environ["REPO"], repo_type="model", token=os.environ["HF_TOKEN"])
    print("\n".join(sorted({"/".join(f.split("/")[:2]) for f in fs if f.count("/") >= 2})))
except Exception as e:
    print("")
PY
)
echo "[已在 HF] $(echo "$HAVE" | grep -c . ) 个目录"

pick_steps() {   # 抽样:全部 <=6 个就都要,否则最早 2 个 + 最后 4 个(覆盖整条弧线)
  local all=($1)
  if [ "${ALL_STEPS:-0}" = "1" ] || [ ${#all[@]} -le 6 ]; then echo "${all[@]}"; return; fi
  echo "${all[0]} ${all[1]} ${all[@]: -4}"
}

for EXP in questa_teacher_v5 questa_teacher_v6 questa_teacher_v7; do
  D=$MS_CKPTS/$EXP
  [ -d "$D" ] || { echo "== $EXP: 目录不存在,跳过"; continue; }
  ALL=$(ls "$D" | grep -oE 'global_step_[0-9]+' | sed 's/.*_//' | sort -n | uniq | tr '\n' ' ')
  [ -n "$ALL" ] || { echo "== $EXP: 无 ckpt"; continue; }
  SEL=$(pick_steps "$ALL")
  echo "== $EXP: 现有 [$ALL] -> 本次传 [$SEL]"
  for S in $SEL; do
    CK=$D/global_step_$S
    [ -d "$CK/actor" ] || { echo "   step $S: 无 actor 分片,跳过"; continue; }
    if echo "$HAVE" | grep -qx "$EXP/step_$S"; then echo "   step $S: HF 上已有,跳过"; continue; fi
    # 磁盘余量守门:合并需要约 2 倍模型大小
    AVAIL=$(df -BG --output=avail "$MS_CKPTS" | tail -1 | tr -dc 0-9)
    [ "${AVAIL:-0}" -lt 30 ] && { echo "   !! 磁盘只剩 ${AVAIL}G,停止合并"; break; }
    OUT=$TMP/$EXP/step_$S
    echo "   step $S: 合并 -> $OUT"
    rm -rf "$OUT"; mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES= nice -n 15 ionice -c3 \
      $PY -m verl.model_merger merge --backend fsdp --local_dir "$CK/actor" --target_dir "$OUT" \
      || { echo "   !! step $S 合并失败,跳过"; rm -rf "$OUT"; continue; }
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json generation_config.json added_tokens.json chat_template.jinja; do
      [ -f "$MS_MODEL/$f" ] && cp "$MS_MODEL/$f" "$OUT/" 2>/dev/null
    done
    echo "   step $S: 上传 $(du -sh "$OUT" | cut -f1)"
    HF_TOKEN=$HF_TOKEN nice -n 15 hf upload "$REPO" "$OUT" "$EXP/step_$S" --repo-type model \
      && rm -rf "$OUT" && echo "   step $S: 完成" \
      || echo "   !! step $S 上传失败,保留 $OUT 以便重试"
  done
done

# 优化器状态(可选):只传 v7 跨越崩溃点的那几个,用于验证有效步长/√v̂ 理论
if [ "${WITH_OPTIM:-0}" = "1" ]; then
  D=$MS_CKPTS/questa_teacher_v7
  for S in ${OPTIM_STEPS:-110 120 130}; do
    CK=$D/global_step_$S
    [ -d "$CK/actor" ] || { echo "== optim step $S: 不存在"; continue; }
    echo "$HAVE" | grep -qx "questa_teacher_v7/step_${S}_fsdp" && { echo "== optim step $S: 已有"; continue; }
    echo "== optim step $S: 上传原始 FSDP 分片(含 Adam m/v)$(du -sh "$CK/actor" | cut -f1)"
    HF_TOKEN=$HF_TOKEN nice -n 15 hf upload "$REPO" "$CK/actor" "questa_teacher_v7/step_${S}_fsdp" --repo-type model \
      || echo "   !! optim step $S 上传失败"
  done
fi
rmdir "$TMP"/*/ "$TMP" 2>/dev/null
echo "[$(date +%H:%M)] 全部结束。日志: $LOG"
