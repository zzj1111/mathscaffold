# mathscaffold: QuestA 复现 + 逐题自适应剂量

三臂,同数据(QuestA 12.5k 难题)、同基座(OpenMath-Nemotron-1.5B)、同算力(8xB200):
- **static**:QuestA 原版全局课程(所有题 r=50,switch_cycle 后 r=25)。纯 hint。
- **adaptive**:逐题 r_q,组构成驱动——全败 +15(上限 90)、全胜 -15(下限 0)、
  混合不动;r=0 后裸探测:成功即毕业,失败回 25;毕业后复发再拉回。纯 hint。
- **teacher**:调查型 teacher(teacherflow,与 ALFWorld/Search 同一工作流)握完整
  四形态决策空间——题级 hint 前缀(ratio_ops)+ 题型级文本(item_ops:skill/
  example/plan,p_ops 剂量,ALFWorld 语义:general 随行、组内同 prompt、构建期
  掷币、记录 text_inj 供注入/裸拆分)。毕业记账保持机械;坏输出回退 adaptive 规则。
  两次真 API 行为测试通过(方向全对;文本条目为读轨迹后针对病灶所写)。

假说:同算力下自适应 > 静态(静态浪费在两处:仍全败的题提示不够浓、已会的题提示
赖着不走)。指标:AIME24/25、MATH500 的 hint-free pass@1(temp0.7/top-p0.95/n16)。

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
