# B200 启动手册(8 卡,两臂)

## 0. 一次性准备
```bash
git clone git@github.com:zzj1111/mathscaffold.git && cd mathscaffold
pip install math_verify datasets    # verl/vllm 用机器上现成的
huggingface-cli download foreverlasting1202/QuestA --repo-type dataset --local-dir data/questa_12k
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir models/OpenMath-Nemotron-1.5B
huggingface-cli download foreverlasting1202/QuestA-Nemotron-1.5B --local-dir models/QuestA-Nemotron-1.5B
export MS_ROOT=$PWD
export MS_MODEL=$MS_ROOT/models/OpenMath-Nemotron-1.5B
export MS_CKPTS=/path/to/ckpts            # TODO: 那台机的 ckpt 根
export MS_DATA="$MS_ROOT/data/questa_12k/OpenR1-25-0-4.jsonl,$MS_ROOT/data/questa_12k/OpenR1-50-0-4.jsonl"
export OPENAI_API_KEY=...                 # teacher 臂需要
export MS_WANDB=1 MS_WANDB_ENTITY=mhong-university-of-minnesota MS_WANDB_PROJECT=mathscaffold
export WANDB_ENTITY=$MS_WANDB_ENTITY WANDB_PROJECT=$MS_WANDB_PROJECT   # 训练器自己的 run 也进同一项目
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

## 3. 两臂开跑(各 30 周期 = 300 步;可 4+4 卡并行或 8 卡串行)
```bash
# 机械臂
MS_EXP=questa_adaptive  MS_WORK=$MS_ROOT/runs/adaptive \
  bash scripts/run_arm.sh adaptive 30 &
# teacher 臂(完整形态)
MS_EXP=questa_teacher   MS_WORK=$MS_ROOT/runs/teacher \
  bash scripts/run_arm.sh teacher 30 &
```
并行时给两臂各设 CUDA_VISIBLE_DEVICES 与 MS_N_GPUS=4。

## 4. 周期探针(可选,建议每 3 周期)
```bash
# 对最新 ckpt 起 vLLM 后:
python scripts/eval_probe.py --base-url http://... --set aime24 > /tmp/p.json
# 把结果写入 $MS_WORK/probe.json({"cycle": N, "aime24": 0.xx, ...}),
# teacher 下周期的开场白会自动带上
```

## 冒烟已在 4xH200 验证的事实(2026-08-14)
- 全链路:parquet→GRPO+vLLM→Math-Verify(线程安全)→recorder→控制器分化(全败+15/全胜-15/混合不动)
- 实测 64 题 × n8:成功率 0.268,组构成 36 混合 / 26 全败 / 2 全胜
- 三个已修的坑:verl 需三个 micro-batch 键;prompt 需 boxed 指令;math_verify 线程环境必须 parsing_timeout=None
