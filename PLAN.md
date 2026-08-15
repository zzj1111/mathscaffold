# mathscaffold: QuestA 复现 + 逐题自适应剂量

两臂(只跑我们自己的方法;QuestA 原版不复现,其论文数字与发布 ckpt 作外部基线),
同数据、同基座(OpenMath-Nemotron-1.5B)、同算力(8xB200)。
数据 = QuestA 两阶段文件合并去重池(8,843 题,qid=题面哈希,不分阶段一池同训)。
评测 n=32 对齐论文协议(AIME24 72.50 / AIME25 62.29 为对表锚点):
- **adaptive**:逐题 r_q,组构成机械规则——全败 +15(上限 90)、全胜 -15(下限 0)、
  混合不动;r=0 裸探测:成功毕业,失败回 25;毕业后复发再拉回。纯 hint。
- **teacher**:调查型 teacher(与 ALFWorld/Search 同一工作流)握完整形态决策空间——
  题级 hint 前缀(ratio_ops)+ general 文本(skill/example/plan + 单一剂量 p)。
  毕业记账机械;坏输出回退 adaptive 规则。三次真 API 行为测试通过。
两臂差 = LLM 判断力(+文本形态)的净值;adaptive 对 QuestA 数字的差 = 逐题自适应
课程本身的净值。(static 复现代码保留未删,仅不排期。)

## 组件(全部本地干跑验证过)
- mathscaffold/data.py       QuestA 式前缀切分 -> verl parquet(逐题 ratio)
- mathscaffold/controller.py 自适应/静态两个控制器(全路径单测)
- mathscaffold/reward.py     verl custom_reward_function:Math-Verify 判分 +
                             逐 rollout {qid, ratio, score} 记录(controller 的输入)
- scripts/prepare_cycle.py   周期边界:读 recorder -> 控制器更新 -> 重生成 parquet
- scripts/run_arm.sh         循环驱动;scripts/train_stage.sh 训练模板(TODO 标注机器路径)
- scripts/eval_probe.py      AIME24/25/MATH500 探针(永远无提示)

## B200 机器上的待办
1. 填 train_stage.sh 的 TODO(verl 路径、模型路径、ckpt 根)。
2. 吞吐烟雾:BS=128 n=16 24k 生成,量一步耗时,定 steps_per_cycle 与总步数。
3. 校准评测:先用已发布的 QuestA-Nemotron-1.5B 跑 eval_probe(AIME24 应约 72.5,
   对不上就先修评测协议再训练)。
4. 数据同步:questa_12k jsonl + Nemotron 基座(或机器上直接重新下载)。
