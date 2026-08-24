# B200 启动手册(8 卡,两臂)

## 0. 一次性准备

### 0a. 独立环境 —— 直接装打包好的离线环境(推荐)
`alfworldauto/auto` 那个 venv 里的 verl 0.3.1.dev 是 ALFWorld agent 分支(main_ppo 无条件
make_envs、assert rollout.n==1、reward manager 只认 episode),跑不了数学,不要复用。

H200 上跑通的整套环境已经冻结成离线 wheelhouse 传到 HF(私有数据集 `Mingyi-Hong/mathscaffold-env`,
6.1 GB,357 个 wheel + verl 源码 + 安装脚本;python 3.12 / torch 2.8.0+cu128(arch 含 sm_100)/
vllm 0.11.0 / flash-attn 2.8.1 / verl v0.7.0+Dr.GRPO 增量 / math-verify 0.9.0;安装脚本在源机上
从零验证过:不碰 PyPI,356 包解析安装 + verl 可编辑 + import 检查全过):
```bash
hf download Mingyi-Hong/mathscaffold-env --repo-type dataset --local-dir /scratch/<you>/msenv_bundle
bash /scratch/<you>/msenv_bundle/install_b200.sh /scratch/<you>/msenv     # ~5 分钟,末尾打印各版本 + cuda arch list
export MS_PYTHON=/scratch/<you>/msenv/bin/python                          # 之后所有脚本都用它,不依赖 PATH/激活
```
解释器与 venv 都落在持久盘(脚本把 `UV_PYTHON_INSTALL_DIR` 放在 venv 旁边),pod 回收不会再出现
`bin/python` 悬空软链静默落到系统 3.10 的情况。

<details><summary>备选:从 PyPI 自己装(仅当 HF 不可达)</summary>

```bash
export UV_PYTHON_INSTALL_DIR=/scratch/<you>/uv-python
uv python install 3.12
uv venv /scratch/<you>/msenv --python 3.12
source /scratch/<you>/msenv/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install vllm==0.11.0
uv pip install flash-attn==2.8.1 --no-build-isolation
git clone --branch v0.7.0 --depth 1 https://github.com/volcengine/verl.git /scratch/<you>/verl-0.7.0
uv pip install -e /scratch/<you>/verl-0.7.0
uv pip install math_verify openai wandb pandas pyarrow datasets huggingface_hub
export MS_PYTHON=/scratch/<you>/msenv/bin/python
```
</details>

### 0b. 仓库、数据、模型、变量
```bash
git clone git@github.com:zzj1111/mathscaffold.git && cd mathscaffold
huggingface-cli download foreverlasting1202/QuestA --repo-type dataset --local-dir data/questa_12k
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir models/OpenMath-Nemotron-1.5B
huggingface-cli download foreverlasting1202/QuestA-Nemotron-1.5B --local-dir models/QuestA-Nemotron-1.5B
export MS_ROOT=$PWD
export MS_MODEL=$MS_ROOT/models/OpenMath-Nemotron-1.5B
export MS_CKPTS=/scratch/<you>/ckpts_math          # 持久盘
export MS_DATA="$MS_ROOT/data/questa_12k/OpenR1-25-0-4.jsonl,$MS_ROOT/data/questa_12k/OpenR1-50-0-4.jsonl"
export OPENAI_API_KEY=...                          # teacher 臂需要(别在共享 tmux 里明文敲,用 AUTOSCAFFOLD_OPENAI_KEY_FILE=<文件> 更稳)
export MS_WANDB=1 MS_WANDB_ENTITY=mhong-university-of-minnesota MS_WANDB_PROJECT=mathscaffold
export WANDB_ENTITY=$MS_WANDB_ENTITY WANDB_PROJECT=$MS_WANDB_PROJECT   # 训练器自己的 run 也进同一项目
wandb login                                        # 一次性
```

## 1. 冒烟(半小时,验证那台机的 verl 键名)
```bash
MS_GPUS=0,1,2,3 MS_N_GPUS=4 MS_MAXRESP=16384 bash scripts/smoke_local.sh
# 通过标准:[SMOKE] PASS 且成功率 ~0.2-0.3、控制器出现分化动作
```

## 2. 评测校准(训练前必做)
```bash
python -m vllm.entrypoints.openai.api_server --model models/QuestA-Nemotron-1.5B \
  --served-model-name actor --port 8100 --tensor-parallel-size 1 &
python scripts/eval_probe.py --base-url http://127.0.0.1:8100/v1 --set aime24
# 应得 ≈0.725(论文 72.50);偏差大先查评测协议,不训练
```

## 3. 开跑(只跑 teacher 臂,8 卡,30 周期 = 300 步)
```bash
cd $MS_ROOT                                   # 必须在仓库根目录(runs/ 相对路径)
source .venv/bin/activate                     # uv 环境 verl;或 export MS_PYTHON=/path/to/verl/.venv/bin/python
export MS_ROOT=$PWD MS_N_GPUS=8               # 上面 0 节的 MS_MODEL/MS_CKPTS/MS_DATA/OPENAI_API_KEY/wandb 也都要在
wandb login                                   # 一次性;没登录的话 nohup 会卡在交互提示上,preflight 会直接报 FAIL
MS_EXP=questa_teacher_b200 bash scripts/launch.sh teacher 30   # 起臂;打印 pid + 日志路径,20 秒后回显前几行
```
`launch.sh` 会自动用**带日期的日志名**:标准输出在 `runs/teacher/logs/teacher_YYYYmmdd_HHMMSS.stdout.log`,
本次启动的 arm.log / train_cN.log / probe.log / watch.log / wandb 本地文件在
`runs/teacher/logs/YYYYmmdd_HHMMSS/`(`runs/teacher/logs/latest` 永远指向最新一次)。重启/续跑不会再
往旧文件里追加,wandb 也不再往仓库目录写(以前 `git pull` 会撞 `wandb/debug.log`,已改)。

`run_arm.sh` 开头有 **preflight**:打印 python/模块/模型/数据/ckpt 根/GPU/OpenAI key/wandb 登录状态,
任一项不满足立即 `[preflight] FAIL: <原因>` 退出(exit 2)。所以 **teacher.log 空 = 进程根本没起来**
(多半是 `nohup` 那行的重定向失败:`runs/` 不存在,或不在仓库根目录),不会再有"静默"情况。

启动后自查(浏览器):wandb 项目 `mathscaffold` 里会出现三条 run——
`questa_teacher`(verl 逐步指标)、`questa_teacher_arm`(每周期组构成/teacher 决策/探针)、
**`questa_teacher_watch`**(Logs 页实时滚动 teacher.log + arm.log;`status/alive` 掉到 0 时
`status/exit_reason` 表里是最后 40 行,`status/last_error` 表收集 Traceback/FAIL/[retry] 行;还有
`gpu/util_mean`、`status/ckpt_step`)。崩了不用 ssh 就能看到怎么崩的。

启动后自查(命令行):
```bash
tail -f runs/teacher.log                      # 依次:preflight OK → [prepare] cycle 0 → verl/Ray/vLLM 启动 → step:1 ...
ls runs/teacher/                              # arm.log / train_c0.parquet / train_c0.log / rollouts_c0.jsonl 逐个出现
nvidia-smi                                    # 2-3 分钟后 8 卡应有显存占用,生成期利用率 50%+
```
续跑(训练进程卡死/被杀后从最近 ckpt 接着来,复用本周期 parquet):
```bash
mv runs/teacher/rollouts_c<N>.jsonl runs/teacher/rollouts_c<N>.stalled.jsonl      # 归档卡住那次的记录器,避免重复计数
MS_START_CYCLE=<N> MS_SKIP_PREPARE=1 MS_EXP=questa_teacher_b200 bash scripts/launch.sh teacher 30
```

常见 FAIL 与处置:
- `python module 'verl' missing` → 没激活 uv 环境;`source .venv/bin/activate` 或 `export MS_PYTHON=...`
- `MS_DATA file not found` / `MS_MODEL dir not found` → 0 节的下载没做或路径不对
- `wandb is not logged in` → `wandb login`(或 `export WANDB_API_KEY=...`);不想传就 `export MS_WANDB=0 MS_TRAINER_LOGGER="['console']"`
- `teacher arm needs OPENAI_API_KEY` → export 它


## QuestA 对齐的优化设置(现为默认)
论文口径(QuestA §5 / AReaL yaml):AdamW 恒定 lr 2e-5;batch 128;`ppo_n_minibatches=1`;
无 KL;clip 0.2;temp 1.0;n=16;生成上限 **32768**(2026-08-22 起默认;24k 是 v1–v3 前期的值,v3 两臂在 24k 上限下 step 60–90 长度坍缩)。

**minibatch 语义,两框架相反,已踩坑(2026-08-21 本地 step150 续训):**
AReaL 的 `ppo_n_minibatches=1` = 整个 batch 作为**一个** minibatch → 每步 **1 次**优化器更新
(最保守的纯 on-policy 单步更新,这也是它能配 lr 2e-5 的前提)。verl 的
`ppo_mini_batch_size=1` = minibatch 为 **1 道题** → 每步 **128 次**更新。按后者跑 30 步的
实测:熵 0.53→0.09、24k 截断率 6%→43%、同剂量正确率 0.42→0.25(v1 同区间稳定在
0.50–0.55)。**现默认 `MS_MINI_BS=128`**(= train batch → 每步 1 次更新,严格等于 QuestA 的 n_minibatches=1),lr 2e-5,
**MS_R0=50**(起始剂量 = 上限,与 QuestA Partial_50 起点一致;teacher 只能按需下调)、MS_R_MAX=50;回退旧值用 MS_LR=1e-6 MS_MINI_BS=32 MS_R0=50 MS_R_MAX=90。
v1 的每步 4 次更新用 MS_MINI_BS=32。

**训练 prompt 格式**:默认 `MS_PROMPT_STYLE=paper` = QuestA 论文附录 B.8 的模板(chat template 内:
题干 + `## Hint.`前缀 + "Please reason step by step, and put your final answer within \boxed{}."),
与 v1 逐字相同。`repo_raw`(公开仓库 add_prefix.py 字面 + 裸文本,不套模板)只作对照——实测让
基座一半生成复读不停,不要用于正式训练。

续跑旧实验示例:
```bash
cat $MS_CKPTS/questa_teacher/latest_checkpointed_iteration.txt   # 例:310 → 下一周期 C=31
MS_START_CYCLE=<C> bash scripts/launch.sh teacher 50
# 周期中途死的场景加 MS_SKIP_PREPARE=1 并沿用该周期号
```
注意:①lr 仍比 v1 高 20 倍,前 1-2 个周期盯 score/熵/截断率(探针每 50 步自动出);
②曲线会呈两段(1e-6 段 / 2e-5 段),wandb 同 run 续写,分析时以切换步为界。

## 3b. 对照臂:QuestA 静态课程(第二个节点跑这个)
`static` 臂 = QuestA 自己的两阶段课程:前半程全体 50% 提示,`MS_SWITCH_CYCLE` 起改为 25%,无 teacher、
无文本 memo、无逐题剂量;优化器/prompt/探针与 teacher 臂完全相同。它同时是 QuestA 的复现和 teacher 的
直接对照。与 teacher 臂的差别只在 `launch.sh` 的臂名和实验名:
```bash
export MS_EXP=questa_static_v3 MS_WORK=$MS_ROOT/runs/static_v3 MS_SWITCH_CYCLE=25   # 50 周期 → 第 25 周期切到 25%
cd $MS_ROOT && bash scripts/launch.sh static 50
```

## 3c. 每周期裸探针(hint-free,训练分布内;默认开)
每个 cycle 训练结束后自动跑 `scripts/bare_probe.sh`:固定的 200 道 **held-out** 题(从训练轮转里永久
剔除)+ 200 道**训练内**题,r=0 无提示、训练采样条件、每题 `MS_BARE_N`(默认 4)条、训练 reward 判分。
8 卡约 5–8 分钟。结果:`$MS_WORK/bare_probe.jsonl`(逐周期)/ `bare_probe.json`(最新),进 teacher
开场白(含 held-out 趋势)和 wandb(`bare/heldout_pass1`、`bare/train_pass1`、`bare/gap_train_minus_heldout`)。
用途:带提示的 score 在涨而裸分不涨 = 模型在学"续写提示"而非解题,teacher 据此退火而不是加剂量;
训练内 − held-out 的差 = 记忆成分。题目集在 `mathscaffold/bare_probe_sets.json`(qid 为题面哈希,两台机
同一份)。关闭:`MS_BARE_PROBE=0`;间隔:`MS_BARE_EVERY=N`。
**中途切换到此版本的 run**:轮转池从 8,843 变成 8,643,切换那一刻轮转会重洗一次(部分题提前/延后
回访,无其他影响);held-out 的 200 题在切换前的周期里可能已被训过一次,分析时注明。

## 3d. 探针服务池(2026-08-21 起两种探针共用 `scripts/vllm_pool.sh`)
起服务前等训练把显存放干净(默认最多 10 分钟,超时打印占用进程)、端口动态分配(同池单调递增)、
每个服务 `setsid` 独立进程组、结束时按进程组整体回收并等显存归零、服务起不来**立即失败并打印
其日志尾部**(exit 3,run_arm 记一行 `[probe]/[bare] failed` 继续训练)。排查顺序:
`runs/<arm>/logs/latest/bare/vllm_c<N>_g<G>.log`(裸探针)或 `runs/<arm>/probe_vllm_c<N>_g<G>.log`
(官方探针)→ 看最后 25 行;常见:`ninja` 缺失(PATH 已自动加 env/bin)、显存未释放(看 `[pool] waiting`
行与其后的占用进程表)、端口被占(自动跳过)。

**正在跑的臂切到新代码**(run_arm.sh 是运行中的 bash,**不能原地 git pull 后指望它生效**,反而会错位):
```bash
# 1) 等周期边界(arm.log 刚出现 "[prepare] cycle N" 之前/之后都行),按会话号整体停臂
SID=<run_arm.sh 的 pid>; kill -TERM $(ps -eo pid,sid | awk -v s=$SID '$2==s{print $1}'); sleep 30; nvidia-smi
# 2) 更新代码
cd $MS_ROOT && git pull && git log --oneline -1
# 3) 从下一周期续跑(ckpt 与剂量状态自动接上;若周期中途被停则加 MS_SKIP_PREPARE=1 并沿用该周期号,
#    并先 mv 掉该周期的半截 rollouts_c<N>.jsonl)
cat $MS_CKPTS/$MS_EXP/latest_checkpointed_iteration.txt      # 例 30 → 下一周期 C=3
MS_START_CYCLE=<C> bash scripts/launch.sh teacher 50           # static 臂同理换臂名
```

## 3e. 回滚到某个周期(不接受后续 teacher 决策)
prepare 每周期会存 `ratio_state_c<N>.json` 快照(2026-08-22 起)。更早的 run 没有快照时用回放:
```bash
cd $MS_ROOT && mv runs/<arm>/ratio_state.json runs/<arm>/ratio_state.pre_rollback.json
$MS_PYTHON scripts/replay_state.py --work runs/<arm> --upto <N> --out runs/<arm>/ratio_state.json
# 用 rollouts_c0..c{N-1} + teacher_transcripts/c0..c{N} 确定性重放 prepare 的状态转移(本地验证与真实状态逐字节一致);
# 同时把 teacher 的 history.json 截到 <= N(旧的留 .pre_replay)
```
之后挪走 `global_step_>N*10` 的 ckpt、把 `latest_checkpointed_iteration.txt` 写成 N*10、归档 `rollouts_c{N}..`、
`train_c{N+1}..`,用 `MS_START_CYCLE=N MS_SKIP_PREPARE=1`(复用 train_c{N}.parquet)续跑;换 `MS_WANDB_RUN_ID`
以免 wandb 丢点。

## 3f. 训练段看门狗(`scripts/stage_watchdog.sh`,2026-08-23 修正)
`run_arm.sh` 每个训练段旁边跑一个看门狗:`train_c<N>.log` 超过 `STALL_MIN` 分钟没变化就杀掉本段、
从最近 ckpt 重试(最多 3 次,然后本臂停止)。verl 只在每步结束才写日志,32K 一步(含段末存 ckpt)
实测 25–37 分钟,旧默认 30 分钟在 2026-08-23 01:2x/01:4xZ 把两臂健康的 79/80、72/73 段误杀,各重走
了一遍 70→80。现默认:`MS_MAXRESP>=32768` → 90 分钟,否则 45;`MS_STALL_MIN` 可覆盖。同时不再误杀
`wandb_watch.py`(它的环境里也有 `MS_EXP`,旧版把它一起杀了 → `<exp>_watch` 面板停更)。

看门狗脚本每次重试都从磁盘重新启动,`git pull` 后**下一段**自动生效,当前正在跑的段仍是旧阈值;
要保住当前段(不重启)可以临时让日志保持"新鲜"直到本段结束(只改 mtime,不改内容):
```bash
nohup bash -c 'for i in $(seq 60); do touch $MS_WORK/logs/latest/train_c*.log; sleep 300; done' >/dev/null 2>&1 &
```
被误杀的 watcher 手动拉起(`--arm-pid` 填 run_arm.sh 的 pid,见 `$MS_WORK/logs/<arm>_<tag>.pid`):
```bash
cd $MS_ROOT && setsid nohup $MS_PYTHON scripts/wandb_watch.py --work $MS_WORK --exp $MS_EXP \
  --stdout-log $(ls -t $MS_WORK/logs/*.stdout.log | head -1) --arm-log $MS_WORK/logs/latest/arm.log \
  --arm-pid $(cat $(ls -t $MS_WORK/logs/*.pid | head -1)) --ckpts $MS_CKPTS >> $MS_WORK/logs/latest/watch.log 2>&1 < /dev/null &
```
重试会沿用同一个 wandb run id:重走的 step 被 wandb 的单调 step 规则丢掉(`Tried to log to step 73
that is less than the current step 79`),曲线到超过旧最高步才续上;`train_c<N>.log` 是全的,死掉那次
的 rollouts 归档为 `rollouts_c<N>.attempt<k>.jsonl`。

## 3g. 从某一步换学习率续训(优化器重建,周期号/步数/剂量状态都接上)
verl 续训会连优化器和 lr 调度器一起恢复,**旧 lr 会被原样带回来**(`MS_LR` 改了也没用)。
`train_stage.sh` 现在自动识别:最近 ckpt 的 `actor/` 里没有 `optim_*`/`extra_state_*` 分片 → 只加载
模型权重,按当前 `MS_LR` 新建优化器与调度器,global step 仍取目录名;之后各段的 ckpt 都带优化器,
照常完整恢复。本地 4 卡冒烟验证:`actor/lr` 从 2e-5 变为 5e-06、step 从 N 继续。做法(以 teacher
臂、从 step 70、lr 5e-6 为例;static 臂把 teacher/teacher_v3 换成 static/static_v3,其余一样):
```bash
# 0) 按会话号整体停臂(run_arm.sh 的 pid 在 $MS_WORK/logs/<arm>_<tag>.pid),等显存归零
SID=$(cat $(ls -t $MS_WORK/logs/*.pid | head -1)); kill -TERM $(ps -eo pid,sid | awk -v s=$SID '$2==s{print $1}'); sleep 60; nvidia-smi
cd $MS_ROOT && git pull && git log --oneline -1
# 1) 新实验名 + 新 wandb run id(旧 run 的单调 step 规则会丢点),其余 MS_* 不变;MS_WORK 沿用(剂量状态/teacher 记忆/裸探针记录接上)
export MS_EXP=questa_teacher_v3c MS_WANDB_RUN_ID=questa_teacher_v3c MS_LR=5e-6
# 2) 用旧 ckpt 的模型分片播种新实验目录(不拷 optim_*/extra_state_*;硬链接不占空间)
OLD=$MS_CKPTS/questa_teacher_v3/global_step_70; NEW=$MS_CKPTS/$MS_EXP/global_step_70
mkdir -p $NEW/actor && cp -l $OLD/actor/model_world_size_*.pt $NEW/actor/ 2>/dev/null || cp $OLD/actor/model_world_size_*.pt $NEW/actor/
cp -r $OLD/actor/huggingface $OLD/actor/fsdp_config.json $NEW/actor/ && cp $OLD/data.pt $NEW/
echo 70 > $MS_CKPTS/$MS_EXP/latest_checkpointed_iteration.txt
ls $NEW/actor    # 应只有 model_world_size_8_rank_*.pt + huggingface/ + fsdp_config.json
# 3) 该周期的半截 rollouts 挪开(否则下次 prepare 重复计数),从该周期续跑、复用已准备好的 parquet
mv $MS_WORK/rollouts_c7.jsonl $MS_WORK/rollouts_c7.attempt_lr2e5.jsonl 2>/dev/null || true
MS_START_CYCLE=7 MS_SKIP_PREPARE=1 bash scripts/launch.sh teacher 50
# 3b) 若该臂在 step N 之后又跑过若干周期(如 teacher 在 lr 2e-5 下跑到了周期 9),先把 MS_WORK 里的
#     周期状态一并退回 N/10 周期,否则 teacher 会继承坍缩期的剂量决策与裸探针读数:
K=7; A=$MS_WORK/archive_lr2e5; mkdir -p $A
cp $MS_WORK/ratio_state.json $A/ratio_state.pre_rollback.json && cp $MS_WORK/ratio_state_c$K.json $MS_WORK/ratio_state.json
$MS_PYTHON - "$MS_WORK" "$K" <<'PY'
import json, sys, os
w, k = sys.argv[1], int(sys.argv[2])
hp = os.path.join(w, "teacher_transcripts", "history.json")
if os.path.exists(hp):
    h = [e for e in json.load(open(hp)) if e.get("cycle", 0) <= k]; json.dump(h, open(hp, "w"), ensure_ascii=False)
bp = os.path.join(w, "bare_probe.jsonl")
if os.path.exists(bp):
    recs = [json.loads(l) for l in open(bp) if l.strip()]; keep = [r for r in recs if r["cycle"] <= k]
    open(bp, "w").write("".join(json.dumps(r) + "\n" for r in keep))
    if keep: json.dump(keep[-1], open(os.path.join(w, "bare_probe.json"), "w"))
print("history/bare trimmed to cycle <=", k)
PY
for c in $(seq $K 20); do for f in rollouts_c$c.jsonl rollouts_c$c.attempt*.jsonl train_c$((c+1)).parquet teacher_transcripts/c$((c+1)).json ratio_state_c$((c+1)).json; do
  [ -e $MS_WORK/$f ] && mkdir -p $A/$(dirname $f) && mv $MS_WORK/$f $A/$f; done; done; ls $A
# 4) 验证:train_c7.log 头部有 "[stage] ckpt global_step_70 has no optimizer shards",第一行 step 指标里
#    actor/lr:np.float64(5e-06);wandb 新 run questa_teacher_v3c 从 step 71 起画
```

## 3h. v4:从 0 起跑,lr 5e-6 + 24K + 软长度惩罚(`scripts/launch_b200_v4.sh`)
lr 2e-5 在四次 run 里都于 65–90 步长度失控(24K 与 32K 各两次),v4 只改两处:lr 5e-6;DAPO 式
overlong buffer——回复超过 `MS_MAXRESP-MS_OVERLONG_LEN`(24000-4096=19904)token 后奖励线性下降,
到上限处为 `-MS_OVERLONG_PENALTY`(默认 0.5),其余样本不受影响,给策略一个"该收尾了"的梯度而不是
撞上限时的零分悬崖。其他与 QuestA yaml 一致:batch 128×n16、一步一更新(mini 128)、R0=50、paper prompt。
```bash
cd /scratch/hongpaul-sandbox/mathscaffold && git pull
export MS_PYTHON=/scratch/<you>/msenv/bin/python WANDB_API_KEY=... OPENAI_API_KEY=...   # 其余路径脚本自带,可用同名变量覆盖
bash scripts/launch_b200_v4.sh teacher      # 节点 1 → runs/teacher_v4, ckpts/questa_teacher_v4, wandb questa_teacher_v4{,_arm,_watch}
bash scripts/launch_b200_v4.sh static       # 节点 2 → 同上 static_v4;MS_SWITCH_CYCLE=25
```
预检:路径、GPU 显存全空、目标目录不存在(只做从零起跑;续跑走 3d)。验证:`train_c0.log` 里
`Training from scratch`、`[stage] overlong penalty on: linear from 19904 to 24000 tokens`、第一步
`actor/lr:np.float64(5e-06)`;wandb 多出 `overlong_reward`/`overlong`(惩罚均值、被罚比例)与 `acc`(未罚的正确率)。
不要惩罚:`MS_OVERLONG_LEN= bash scripts/launch_b200_v4.sh teacher`;换力度:`MS_OVERLONG_PENALTY=1.0`(DAPO 原值)。
实现:`train_stage.sh` 的 `MS_OVERLONG_LEN/MS_OVERLONG_PENALTY` → `reward_model.reward_manager=dapo` +
`reward_model.reward_kwargs.overlong_buffer_cfg.*`(agent-loop 的 reward loop 走同名 DAPO manager,
自定义 reward 照常被调用;`rollouts_c*.jsonl` 记的仍是未罚的 0/1,teacher 的统计不受影响)。

## 3i. SAGE-bench:在 SAGE(arXiv 2602.03143)的数据/模型上跑我们的 auto-scaffold
一键脚本 `scripts/launch_sagebench.sh`(交给跑实验的人即可):Qwen3-4B-Instruct-2507 +
OpenR1-Math-220k 过滤 15k(带 R1 轨迹,前缀 hint 可用),GRPO 设置对齐 SAGE 的 run 脚本:
lr 1e-6、batch 128×n4、mini 64(每步 2 次更新)、clip 0.2/0.28、KL 0、响应 8K、500 步;
hint 机制 = 我们的 teacher 原样(前缀剂量 R0=50 + 文本 scaffold/skills)。SAGE 自报(6 个
benchmark 均值):GRPO +2.9 / SAGE +4.2(Qwen3-4B)。
```bash
cd <repo> && git pull
export MS_PYTHON=/path/to/env/bin/python WANDB_API_KEY=... OPENAI_API_KEY=...
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
$MS_PYTHON scripts/prepare_sage_data.py --model models/Qwen3-4B-Instruct-2507 \
    --out data/sage15k/openr1_sage15k.jsonl     # 94k → Math-Verify 通过 + 轨迹<8192 token → 抽 15k;同时生成裸探针题集
bash scripts/launch_sagebench.sh teacher 50     # wandb: sagebench_teacher_qwen3_4b{,_arm,_watch}
```
官方探针改为 SAGE 协议:aime24/aime25/math500/amc23,temp 0.6 / top-p 0.95 / 8K,每 5 周期;
裸探针用该数据集自己的 200+200 题集(`MS_BARE_SETS`,prepare 脚本已生成并打印路径)。与论文数字
对照是近似的(15k 抽样与其不完全一致、我们评 4 个集合、n=32 而非其单次生成);同机自跑一条
无 hint GRPO 基线(`MS_R0=0 MS_TEACHER=off` 另开臂)才是干净对照,需要时再加。
注意:数据行里 answer 需字面出现在 R1 轨迹 </think> 之后的部分(加载器规则),prepare 会打印
因此被丢的行数;prompt 模板沿用我们的 QuestA paper 版(这属于我们方法的一部分)。

## 4. 自动探针(已内置,无需手动)
`run_arm.sh` 每 `MS_PROBE_EVERY`(默认 5)个周期 = 每 50 步,在周期边界自动:
合并最新 ckpt → vLLM 起服务 → 无提示评测 **AIME24 / AIME25 / HMMT25**(各 30 题,
n=32,temp 0.7 / top-p 0.95,Math-Verify)→ 写 `$MS_WORK/probe.json` → 关服务。
结果自动进 teacher 下周期开场白与 wandb(`probe/aime24` 等)。
可调:`MS_PROBE_SETS=aime24,aime25,hmmt25,math500`、`MS_PROBE_N=32`、`MS_PROBE_EVERY=5`。
手动跑某个 ckpt:`MS_EXP=questa_teacher bash scripts/probe_ckpt.sh <cycle>`。
论文锚点(QuestA-Nemotron-1.5B,pass@1@32):AIME24 72.50 / AIME25 62.29 / HMMT25 41.67。

## 冒烟已在 4xH200 验证的事实(2026-08-14)
- 全链路:parquet→GRPO+vLLM→Math-Verify(线程安全)→recorder→控制器分化(全败+15/全胜-15/混合不动)
- 实测 64 题 × n8:成功率 0.268,组构成 36 混合 / 26 全败 / 2 全胜
- 三个已修的坑:verl 需三个 micro-batch 键;prompt 需 boxed 指令;math_verify 线程环境必须 parsing_timeout=None
