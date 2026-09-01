# nanocode 的竞争位置：不卖循环，卖证据

## 结论

nanocode 不该和其他项目比“谁用更少行代码写出 ReAct 循环”，也不该把一次 leaderboard
分数当护城河。最小循环已经商品化：`mini-swe-agent` 把 agent class 压到约 100 行，仍能在
公开基准上取得很高分。公开编码基准本身又不断暴露污染、坏题和基础设施噪声。

本项目最有机会占住的位置是：

> **一个能告诉工程师“哪个 agent 部件真的值回质量、token 和失败风险”的可审计实验型智能体。**

核心资产不是九个工具，而是因果消融、运行前预测、反例任务、完整轨迹和允许证据推翻作者理由
的记录。`DECISIONS.md` 应继续是首页最重要的入口。

## 为什么这个位置有空间

1. **简单 agent 已经很多。** 再加 planner、memory、subagent 或 MCP，只会变成又一个功能清单；
   没有对照实验，没人知道新增部件改善了什么。
2. **单一 leaderboard 越来越难取信。** OpenAI 先后指出 SWE-bench Verified 的测试与污染问题，
   后续又在 SWE-Bench Pro 中估计约 30% 任务有问题。追一个总分不是稳定资产。
3. **基础设施本身能移动几个百分点。** Anthropic 报告 Terminal-Bench 资源配置造成 6 个百分点
   差异，并观察到最高约 6% 的 pod 错误。保存配置、区分模型失败与基础设施失败不是装饰。
4. **真实产品决定是成本—质量决定。** 工程师更常问“这个工具是否值得默认开启”，而不是
   “某个模型在某榜多 1 点”。本项目的搜索消融正好能形成一个可执行的决策案例。

## 应该形成的四层护城河

### 1. 可信实验层

- 每个结论都给 estimand、预注册门槛、功效、完整 artifact 和机器可读 verdict。
- 结果只能是 `PASS / FAIL / INVALID`；缺字段、dirty 代码、基础设施错误超限时失败关闭。
- 把“旧理由—反事实条件—真实结果—现在的决定”做成稳定数据结构，而不只是一篇叙事文档。

### 2. 私有、抗污染任务层

- 新建未公开解答的 repo-scale 任务，加入 canary 和创建时间；公开任务只用于开发 harness。
- 同一能力覆盖小仓库、中仓库和真实依赖图，测出工具价值随规模变化的曲线。
- 每题同时保存需求、行为级 grader、核心/安全标签与多种合法实现，避免只认 gold patch。

### 3. 跨模型决策层

- 在至少一个小模型、一个前沿模型、两个 provider 上复现实验。
- 报告 `模型 × 部件` 交互：搜索可能对弱模型有害、对强模型有益，不能把一个平均数写成规律。
- 输出默认策略，而不是只输出论文表格，例如“小仓库默认隐藏搜索，大仓库按仓库规模开放”。

### 4. 可采用的产品层

- 一条命令生成 manifest、schedule、轨迹、patch、grader 和 verdict bundle。
- 提供静态 HTML/JSON 报告，让评审能从总体效应下钻到具体失败轨迹。
- 给外部 agent 提供 adapter；实验系统能审计别人的 scaffold，受众就不再限于 nanocode 用户。

## 接下来的优先级

| 优先级 | 工作 | 成功标准 |
|---|---|---|
| P0 | 完成搜索确认实验 | 预注册裁决能独立返回 PASS/FAIL/INVALID，结果包可复算 |
| P1 | 仓库规模实验 | 至少三个规模层，每层有新任务；估计搜索价值随规模的变化 |
| P2 | 跨模型复制 | 至少三种固定 snapshot；报告模型与工具的交互区间 |
| P3 | 发布审计包 | 外部项目能用 adapter 跑自己的两个条件，并生成同格式报告 |
| P4 | 再做 agent 功能 | 只有新功能对应的机制指标、反事实和停止规则写定后才加入 |

P1 的外部数据入口已经建立：6 个第三方仓库、26 个 snapshot 后候选 PR，带冻结 SHA、许可证、
patch 哈希和隐藏测试物化器。首轮修正后审计得到 7 个 base-red / gold-green 有效任务；其余 19 个
连同失败原因保留，而不是事后换题。下一门槛是在冻结的 Linux/依赖镜像中裁决 15 个环境错误，
并扩大有效仓库覆盖面；当前 7 题只来自 Click 与 Rich，不能包装成广泛外部效度。

## 明确不做什么

- 不用公开 SWE-bench 总分作为唯一卖点。
- 不把更多工具、更多框架集成或更多 agent 数量本身当创新。
- 不把 12 个目的性任务外推成“所有编程任务”。
- 不隐藏推翻作者原理由的结果；这是项目最稀缺的可信度资产。

## 外部依据

- mini-swe-agent: <https://github.com/SWE-agent/mini-swe-agent>
- OpenAI, SWE-bench Verified 的污染与测试问题：
  <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- OpenAI, SWE-Bench Pro 审计：
  <https://openai.com/index/separating-signal-from-noise-coding-evaluations/>
- Anthropic, agentic coding eval 的基础设施噪声：
  <https://www.anthropic.com/engineering/infrastructure-noise>
