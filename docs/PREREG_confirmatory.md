# 第十二轮预注册：确认性审计

状态：**尚未运行真实模型。** 本文件必须在任何确认性数据产生前提交。

本轮不是替旧结果补显著性，而是修复第八至第十一轮暴露出的识别问题：固定条件顺序、
复合消融、校准探针兼任评分、崩溃轨迹补零、未保存完整轨迹，以及未定义总体却报告
task-bootstrap 区间。

## 已经看过什么

已经看过 `results/ablation.json`、`results/termination.json` 和
`results/architecture.json` 的全部结果。因此本轮不把旧数据导出的方向当成新预测，
也不把同一批数据上的重分析叫确认。

旧数据已经确定的更正：

- 终止 2×2 的正确率交互为 +5.6 点 [−11.1, +22.2]，完成率交互为
  +5.6 点 [−27.8, +41.7]；没有超可加证据。
- Plan-Execute 每条漏计一次规划调用；更正后差为 +2.9 [−0.4, +6.4]。
- 原功效脚本是齐性 Bernoulli 敏感性计算，不是实际设计的校准功效分析。

## estimand：先限定能回答的范围

### 固定任务集 estimand

主 estimand 是：**在本轮明确列出的固定任务、固定模型和固定运行环境上，跨模型随机 seed
的平均处理效应。** 随机化单位是 task × repetition 块内的条件分配顺序。

区间描述模型重复与条件顺序的不确定性，不外推到“所有编程任务”。固定任务集结果必须逐任务
同时报告，不能只给一个总体均值。

### 任务总体 estimand

本轮不声称任务总体效应。若要外推，必须另建有抽样框、来源独立的新任务集；作者目的性写出的
12 个任务不能靠 bootstrap 变成随机样本。

## 共同实验协议

- 条件在每个 task × repetition 块内随机排序；随机种子与完整 schedule 写进结果文件。
- 使用相同 task、rep、模型参数与 seed 的配对条件。
- 有模块补丁的消融每条轨迹在独立进程运行；条件不会共享猴子补丁状态。
- 每条轨迹保留完整 trace、最终 unified patch、评分结果和异常；崩溃不再把已花调用/token 记 0。
- 顶层 manifest 保存 git SHA、dirty 状态、命令、参数、Python/依赖版本、时间和脱敏端点族。
- 正确率使用 `EVAL_SEED=7` 的 40 个 held-out probes，并强制加入所有 discriminating inputs；
  `CALIB_SEED=0` 不再兼任最终评分。
- 基础设施错误单列，不当作模型错误，也不静默删除。任何条件若基础设施错误超过 5%，整轮作废。
- 不边跑边看条件结果；先完成全部 schedule，再执行分析命令。

## A：终止机制的 2×2 确认

条件保持：`old_env_no_budget`、`path_fix`、`budget`、`both`。

主要 estimand 是差中差：

```text
interaction = both - path_fix - budget + old_env_no_budget
```

主要指标按顺序只有两个：

1. 每条轨迹模型调用数；
2. 每条轨迹总 token。

安全性指标：held-out 正确率。完成率、工具调用、命令数和失败命令是机制诊断，不参与主要结论。

判断规则：只有 interaction 自身的 95% 区间排除 0，才能说存在交互。不得再用“合并显著而
单独不显著”替代交互检验。

复现分析：

```bash
python experiments/analyse.py results/termination_confirmatory.json \
  --baseline old_env_no_budget --interaction path_fix budget both --fixed-tasks
```

## B：错误恢复拆分

条件是 2×2：

| 条件 | JSON 解析恢复 | dispatch 前参数恢复 |
|---|---:|---:|
| `full` | 开 | 开 |
| `no_parse_recovery` | 关 | 开 |
| `no_argument_recovery` | 开 | 关 |
| `no_recovery` | 关 | 关 |

主要报告分成两件事：

1. 自然触发率：每类错误在完整条件下多常见；
2. 触发后的损失：崩溃、最终 held-out 正确率、真实调用/token。

不得把 `no_recovery` 的崩溃率当发现；它仍是操纵检验。只有拆开的主效应与交互才能归因到
具体恢复层。若某一错误类型自然触发少于 10 条轨迹，该层只报告“未充分暴露”，不解释 null。

运行时显式指定四条件，避免与历史四条件混淆：

```bash
python experiments/ablation.py \
  --conditions full no_parse_recovery no_argument_recovery no_recovery \
  --out results/recovery_confirmatory.json
```

## C：搜索的成本—质量决策

随机配对 `full` 与 `no_search`。

确认性成本指标：总 token、工具调用和失败命令。质量指标是 held-out 正确率及 graded verifier；
成本改善不能自动推出质量等效。

### 运行前决策门槛

正确率的绝对非劣界预先固定为 **δ = 5 个百分点**。令

```text
Δquality = accuracy(no_search) - accuracy(full)
```

只有固定任务配对差的双侧 95% 区间下界严格高于 `−0.05`，才判 `no_search` 在正确率上
非劣；点估计相等、区间包含 0 或成本显著下降都不能替代这个判断。`graded verifier` 的总体
通过比例也必须用同一界值通过非劣判断，并逐任务报告；任一质量指标不过界，都不执行移除决定。

5 点表示最多容忍每 20 个同类任务多失败 1 个。它不是从旧结果的 ±11 点区间倒推出来的：
第八轮在揭晓结果前已经把预期的近零范围写成 `|Δ| < 5` 点，因此沿用 5 点避免事后放宽。
2 点（每 50 个任务多失败 1 个）比项目原先写下的可忽略范围更严格；10 点则会容许每 10 个
任务多失败 1 个，超过本项目愿意用成本节省交换的质量损失。界值按这个产品损失尺度选择，
不按现有样本量能否让实验过关来选择。

决策分三类：

1. 区间下界 `> −0.05`，且 `Δtoken = token(no_search) - token(full)` 的 95% 区间上界 `< 0`：
   只对本轮同尺度小仓库默认移除搜索；
2. 质量非劣但成本优势未确认，或成本下降但质量未过非劣界：不改默认配置，报告未决；
3. 正确率区间下界 `≤ −0.05`：不得声称非劣，不论节省多少 token。

延迟只作部署指标，不与正确率折成一个事后可调的综合分数。这个门槛只适用于本轮固定的
小仓库任务；大仓库必须另行预注册界值和任务集。

外部材料只约束**怎么选和怎么判**，不替本项目生成 5 点这个数：FDA 的非劣效指导要求界值
代表可接受的最大损失，并用置信区间排除超过界值的损失；NIST 把 δ 定义为真正关心检测的
比例变化，样本量应服从 δ、显著性和功效。编码评测还会受坏题与基础设施错误影响，所以本轮
保留双侧 95% 区间、逐任务 verifier 和基础设施错误护栏。

- FDA: <https://www.fda.gov/media/78504/download>
- NIST: <https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm>
- OpenAI, SWE-bench Verified: <https://openai.com/index/introducing-swe-bench-verified/>
- Anthropic, infrastructure noise: <https://www.anthropic.com/engineering/infrastructure-noise>

### C 的冻结配置

- 模型：`openai:gpt-4o-mini-2024-07-18`，不用会漂移的 `gpt-4o-mini` 别名；这是历史实验
  同一模型家族目前公开的固定 snapshot。
- 任务：`t01_remove_outliers`、`t02_top_k`、`t03_chunk`、`t04_normalize`、
  `t05_merge_configs`、`t06_parse_range`、`t07_dedupe`、`t08_round_price`、
  `t09_split_name`、`t10_moving_average`、`t11_truncate`、`t12_sort_scores`。
- 每任务 84 次重复、两条件，共 1008 个配对 block / 2016 条轨迹。
- `temperature=1.0`、每条 `seed=700+rep`、`max_steps=18`、每次调用最大输出 4096 token、`workers=8`、
  `schedule_seed=20260901`；使用缺一句约束的原始提示，不加 `--unambiguous`。
- 评分固定为 `EVAL_SEED=7`、40 个 held-out probes 加全部 discriminating inputs。
- 旧单价下预计约 $11.7、约 3.6 万次模型调用。开始前必须确认端点配额足以一次完成并预留
  $25；不使用“看到已花费用后中途停止”的规则。配额/网络造成任一条件基础设施错误率超过
  5%，整轮 `INVALID`。

样本量不是按能否承担选择。旧 36 个配对 block 有 6 个不一致，`q=0.167`，Wilson 95%
上界为 0.319。用上界、真实差为 0、δ=0.05、双侧 95% 区间和 80% power，解析近似需要
1002 个 block；取整为 12 × 84 后，按实际“固定任务、任务内重采样”规则模拟的 power 为
0.807（300 次试验、每次 300 次 bootstrap）。复算命令：

```bash
py -3 experiments/power.py --noninferiority --trials 300 --ni-boots 300
```

工作树提交并保持 clean 后，C 唯一允许的真实运行命令是：

```bash
py -3 experiments/ablation.py \
  --model openai:gpt-4o-mini-2024-07-18 \
  --conditions full no_search --reps 84 --max-steps 18 --max-tokens 4096 \
  --temperature 1.0 --workers 8 \
  --tasks t01_remove_outliers t02_top_k t03_chunk t04_normalize \
          t05_merge_configs t06_parse_range t07_dedupe t08_round_price \
          t09_split_name t10_moving_average t11_truncate t12_sort_scores \
  --schedule-seed 20260901 --out results/search_confirmatory.json \
  --require-clean --require-model-snapshot --fail-if-output-exists
```

唯一确认性裁决命令是：

```bash
py -3 experiments/analyse.py results/search_confirmatory.json \
  --baseline full --treatment no_search --fixed-tasks \
  --noninferiority-margin 0.05 --infra-threshold 0.05 \
  --expected-model openai:gpt-4o-mini-2024-07-18 --expected-reps 84 \
  --expected-schedule-seed 20260901 --expected-max-steps 18 \
  --expected-max-tokens 4096 --expected-temperature 1.0 \
  --expected-tasks t01_remove_outliers t02_top_k t03_chunk t04_normalize \
                   t05_merge_configs t06_parse_range t07_dedupe t08_round_price \
                   t09_split_name t10_moving_average t11_truncate t12_sort_scores \
  --decision-json results/search_confirmatory_decision.json
```

裁决器退出码：`0=PASS`、`2=FAIL`、`3=INVALID`。它要求 clean manifest、完整 token 与
graded 字段，逐条件检查基础设施错误率，只使用完整配对 block。

## D：架构实验暂停

旧任务对 gpt-5.5 触顶，继续增加重复没有信息量。ReAct vs Plan-Execute 只有在以下条件全部满足
后恢复：用最终模型重新校准难度、加入不触顶的 held-out 仓库任务、两臂随机交错、规划调用计入
总调用。旧 token 成本结果保留为探索性证据，不据此声称架构质量优劣。

## 样本量与停止规则

- 先用 fixture 做 1 task × 1 rep × 全条件的 harness 冒烟；不看真实模型结果。
- C 的固定任务 estimand 使用上面冻结的 84 次重复；它不因此获得任务总体外推能力。
- A、B 尚未冻结样本量，不得借用 C 的 84 次重复；各自要用对应触发率与主要指标另算。
- 不按已花费用中途停止。若预算或 API 配额不足以完成全部 schedule，本轮不开始；意外中断时
  未完成 block 整体排除，但只有基础设施错误护栏仍满足时才可分析。
- 任何中途代码修复都会生成新 git SHA；修复前后的数据不得合并为同一确认性实验。

## 功效报告

`experiments/power.py --sensitivity` 必须同时报告一类错误率、任务异质性和配对依赖假设。
基线接近 1 时报告“可实现范围内达不到目标 power”，禁止再写“>70 点 MDE”。

功效分析服务于运行前设计，不用于把运行后的 null 改写成零效应或“设计保证”。
