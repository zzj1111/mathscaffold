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
mkdir -p runs
MS_EXP=questa_teacher MS_WORK=$MS_ROOT/runs/teacher \
  nohup bash scripts/run_arm.sh teacher 30 > runs/teacher.log 2>&1 &
sleep 20; cat runs/teacher.log                # 30 秒内必须看到 [preflight] ... OK — entering cycle loop
```
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
常见 FAIL 与处置:
- `python module 'verl' missing` → 没激活 uv 环境;`source .venv/bin/activate` 或 `export MS_PYTHON=...`
- `MS_DATA file not found` / `MS_MODEL dir not found` → 0 节的下载没做或路径不对
- `wandb is not logged in` → `wandb login`(或 `export WANDB_API_KEY=...`);不想传就 `export MS_WANDB=0 MS_TRAINER_LOGGER="['console']"`
- `teacher arm needs OPENAI_API_KEY` → export 它

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
