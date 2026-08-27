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
#   HF_TOKEN=hf_xxx STEPS_questa_teacher_v7="50 100 110 120 130 134" bash upload_ckpts.sh   # 覆盖点名
set -u
REPO=${REPO:-Mingyi-Hong/mathscaffold-ckpts}
# token:优先环境变量,否则用 `hf auth login` 的缓存(不需要在命令行里贴密钥)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN=$(tr -d "[:space:]" < "$HOME/.cache/huggingface/token")
fi
[ -n "${HF_TOKEN:-}" ] || { echo "没有 token:先跑一次 \`hf auth login\`,或 export HF_TOKEN=hf_..."; exit 2; }
# 路径:默认取 launch_b200_v9.sh 里的同一套,可用环境变量覆盖
MS_ROOT=${MS_ROOT:-/scratch/hongpaul-sandbox/mathscaffold}
MS_CKPTS=${MS_CKPTS:-$MS_ROOT/ckpts}
MS_MODEL=${MS_MODEL:-$MS_ROOT/models/OpenMath-Nemotron-1.5B}
[ -d "$MS_CKPTS" ] || { echo "MS_CKPTS 不存在: $MS_CKPTS — 用 MS_CKPTS=... 覆盖"; exit 2; }
PY=${MS_PYTHON:-$(command -v python3 || command -v python)}
TMP=${TMP_MERGE:-$MS_CKPTS/_upload_tmp}
LOG=${LOG:-$HOME/upload_ckpts_$(date +%m%d_%H%M).log}
exec > >(tee -a "$LOG") 2>&1
UP=$(command -v hf >/dev/null 2>&1 && echo hf || echo huggingface-cli)
echo "[$(date +%H:%M)] uploader=$UP repo=$REPO  ckpts=$MS_CKPTS  log=$LOG"

# 已在 HF 上的路径,用于跳过
export REPO HF_TOKEN   # 下面的 python 通过环境变量读它们(漏了 export 会让"已传过就跳过"静默失效)
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

# ---------------------------------------------------------------------------------
# 明确点名,不做启发式抽样。每一步都是从各臂的指标曲线上挑的(wandb 全量 history):
#
# v5  lr 2e-5 + KL loss —— "长度先失控,梯度后爆炸"
#   20 截断 9.6% 已在爬升、score 0.568 仍健康        (失控起点)
#   40 截断 18.6%、len 15.5K,score 0.578 居然还好    (长度已失控但功能正常 <- 关键点)
#   50 截断 14.6%、len 回落 12.8K、score 0.608        (最后一个可用点)
#   60 grad_norm 6.70 = 397x、截断 35%、score 0.209   (爆炸当中)
#   70 截断 97.1%、len 24K、熵 0.181、score 0.000     (已死)
#
# v6  lr 2e-5 + 长度惩罚 + hint=0 —— "长度被摁住,但慢性中毒"
#   20 score 0.419 峰值、截断 0.1%                    (惩罚生效,长度全程不失控)
#   40 score 0.333                                    (下滑中段)
#   50 score 0.317、grad_norm 开始抖动                (下滑后段)
#   60 score 0.249、grad_norm 2.5x                    (终点,始终没有爆炸)
#
# v7  lr 1e-5 + 长度惩罚 —— "长期健康后突然雪崩"(记录最完整的一次)
#   50  官方探针最好点 65.1/53.2/33.4                 (健康基准)
#  100  gn 1.6x、截断 1.9%、score 0.365               (最后一个完全健康点)
#  110  gn 3.1x                                       (第一次异动)
#  120  gn 19.6x、截断 7.0%、score 0.268              (转折)
#  130  gn 456x、截断 97.4%、score -0.469             (雪崩后)
STEPS_questa_teacher_v5="20 40 50 60 70"
STEPS_questa_teacher_v6="20 40 50 60"
STEPS_questa_teacher_v7="50 100 110 120 130"
# 优化器状态(Adam m/v):只在"有效步长理论"真正需要判读的转折点上取。
#   v5 50->60: grad_norm 平了 30 步然后 397 倍爆炸 —— 这期间 sqrt(v) 在漂吗?
#   v7 100->110->120: 健康 -> 异动 -> 转折,最干净的一组跨越
# (v3 的 50/60/70/80 分片含优化器状态,已经在本地,不必从 B200 传)
OPTIM_questa_teacher_v5="50 60"
OPTIM_questa_teacher_v7="100 110 120"

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
    HF_TOKEN=$HF_TOKEN nice -n 15 $UP upload "$REPO" "$OUT" "$EXP/step_$S" --repo-type model \
      && rm -rf "$OUT" && echo "   step $S: 完成" \
      || echo "   !! step $S 上传失败,保留 $OUT 以便重试"
  done
done

# 优化器状态(可选):只传 v7 跨越崩溃点的那几个,用于验证有效步长/√v̂ 理论
if [ "${WITH_OPTIM:-0}" = "1" ]; then
  for EXP in questa_teacher_v5 questa_teacher_v7; do
    eval "OS=\${OPTIM_$EXP:-}"
    for S in $OS; do
      CK=$MS_CKPTS/$EXP/global_step_$S
      [ -d "$CK/actor" ] || { echo "== optim $EXP/$S: 不存在"; continue; }
      echo "$HAVE" | grep -qx "$EXP/step_${S}_fsdp" && { echo "== optim $EXP/$S: 已有"; continue; }
      echo "== optim $EXP/$S: 上传原始 FSDP 分片(含 Adam m/v)$(du -sh "$CK/actor" | cut -f1)"
      HF_TOKEN=$HF_TOKEN nice -n 15 $UP upload "$REPO" "$CK/actor" "$EXP/step_${S}_fsdp" --repo-type model \
        || echo "   !! optim $EXP/$S 上传失败"
    done
  done
fi
rmdir "$TMP"/*/ "$TMP" 2>/dev/null
echo "[$(date +%H:%M)] 全部结束。日志: $LOG"
