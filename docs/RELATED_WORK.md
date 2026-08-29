# 相关工作检索结果（2026-08-28）

**结论先行：这个方向已经饱和。本项目的每一个组件都能在已发表工作里找到对应，
包括最核心的那个「负结果」——它在七周前被发表，命名为 semantic collapse，
并且用比我们大得多的样本量在标准基准上量化过。**

检索范围：arXiv cs.SE / cs.CL、ACM DL、ICSE/FSE 会议页。角度包括：歧义需求与澄清、
约束删除构造、跨模型一致性与相关误差、self-consistency 失效、欠规范代码生成、
仓库级上下文。

---

## 一、逐条对照

| 本项目做的 | 已发表的对应工作 | 判定 |
|---|---|---|
| **BSE**：采样 k 份实现 → 按行为聚类 → 类数 ≥2 就提问 → 用区分输入生成问题 | **ClarifyGPT**（FSE / PACMSE 2024，[arXiv 2310.10996](https://arxiv.org/abs/2310.10996)）：「marks requirements as ambiguous whenever an LLM's interpretation contains at least two **semantic clusters of programs**, and asks clarifying questions to differentiate programs from these clusters」 | **完全重合** |
| 差分测试驱动的歧义检测与修复 | **SpecFix**（[arXiv 2505.07270](https://arxiv.org/abs/2505.07270)）：LLM + 差分测试自动修复歧义描述；修改了 43.58% 的描述，Pass@1 提升 30.9% | **重合** |
| 「删掉恰好一句约束」构造反事实歧义任务集 | **ClarifyCodeBench**（[arXiv 2607.00711](https://arxiv.org/pdf/2607.00711)）：deletion-only editing，标注者删除决定行为所必需的信息 | **重合** |
| 歧义下的采样多样性 / 跨模型一致性测量 | **Orchid**（[arXiv 2604.21505](https://arxiv.org/html/2604.21505v1)）：1304 个函数级歧义任务，intra-model conflict 14.09%→28.29%，inter-model conflict 57.28% | **重合，且结论相反**（见下） |
| **模型在欠规范下收敛到单一错误读法，使得基于分歧的检测失效** | **[arXiv 2607.01953](https://arxiv.org/abs/2607.01953)**（2026-07）《Underspecification does not imply Incoherence: The Risks of Semantic Collapse in Coding Models》：命名为 *detrimental semantic collapse*，MBPP >10%、HumanEval 3%、LiveCodeBench 32%；明确指出这「exposes a fundamental blind spot in disambiguation and correctness estimation techniques that rely on incoherence as a proxy for prompt underspecification」 | **完全重合，且更强** |
| 一致性 = 共享先验 → 跨厂商相关误差 | **Correlated Errors in LLMs**（[arXiv 2506.07962](https://arxiv.org/pdf/2506.07962)）：「models that err tend to err together — on the same answers, more so as they scale, across vendors and architectures; N models agreeing is not N independent confirmations」 | **完全重合**，连「越强的模型越严重」都对上了 |
| 一致性作为置信信号不可靠 | [2607.08065](https://arxiv.org/html/2607.08065)（审计 self-consistency 与跨模型一致性）、[2605.29800](https://arxiv.org/html/2605.29800v1)（九个评委只有两票有效）、[2608.11403](https://arxiv.org/abs/2608.11403)（majority vote 在难题上倒退）、[2608.18795](https://arxiv.org/abs/2608.18795)（wrong-consensus 分解） | **重合** |
| 约束类型分类学（哪类约束被漏掉） | [2604.24703](https://arxiv.org/pdf/2604.24703)《Defective Task Descriptions》等：已列出 numeric bounds / ordering rules / preconditions / edge-case behavior / error handling / output format | **重合** |
| 欠规范提示的行为 | **What Prompts Don't Say**（ACL Findings 2026，[arXiv 2505.13360](https://arxiv.org/abs/2505.13360)）：LLM 默认推断未指定需求成功率 41.1%，欠规范提示在模型更新时回归概率是 2 倍 | **重合** |

---

## 二、一个需要解释的矛盾

**Orchid 与本项目的测量方向相反。**

- Orchid：歧义使 intra-model conflict rate 从 14.09% 涨到 28.29%，「models produce up to
  five distinct functional variants for single ambiguous tasks」
- 本项目：删掉一句约束后，6 份采样的平均行为类只有 **1.08**，几乎不变

可能的原因，全部未验证：

1. **歧义类型不同。** Orchid 注入的是**语言学歧义**（lexical / syntactic / vagueness）；
   本项目删除的是**行为约束句**。前者可能确实引发表面分歧，后者被先验直接填平。
2. **度量不同。** conflict rate 是成对功能差异比例；本项目用的是行为等价类数。
3. **任务池不同。** n=12 对 n=1304。

而 2607.01953 的立场更接近本项目：欠规范**不**蕴含不一致。所以这不是一个孤立的矛盾，
而是一个领域内尚未解决的分歧——**什么时候歧义会引发分歧，什么时候会被先验填平**。

---

## 三、还剩什么（每一条都要保持怀疑）

**a. 符号翻转本身。** 「一致性精度随规范完整度从 1.00 翻到 0.00」这个受控对照，
在检索里没找到直接对应。但在 2607.01953（语义坍缩）+ 2506.07962（相关误差）之后，
审稿人很可能认为它是推论。**弱。**

**b. 「那到底什么能检测 semantic collapse？」** 2607.01953 指出了盲点但（据检索）没有解决它。
这是最自然的后续问题——**也正因为自然，大概率已经有资源充足的组在做。** 学生入场风险高。

**c. 仓库约定作为缺失约束的来源。** 检索显示仓库级工作集中在
*检索/知识图谱提升正确性*（[2601.00376](https://arxiv.org/pdf/2601.00376)、
[2510.04905](https://arxiv.org/html/2510.04905v1)、A³-CodGen 等），
**没有找到「用代码库既有约定来补齐提示遗漏的行为约束」这个具体问法**。
这是目前看起来最可能真空的一块，而且它正好被 2607.01953 的结论所激励：
信息不在模型里，那在不在仓库里？**本项目已建好的执行沙箱与覆盖率校验可以直接复用。**

**d. 成本 / 效率 Pareto（原调研里的 G5）。** 与本方向无关，学术空白仍大，竞争低。

---

## 四、这次检索对项目的意义

不必粉饰：**原选题应当放弃，这个方向不该进。**

但两件事仍然成立：

1. **本项目独立复现了一个七周前才发表的结果。** 从完全不同的路径、不同的构造方法、
   不同的度量出发，得到与 semantic collapse 一致的结论，并且自己就推导出了
   「基于分歧的检测因此失效」这一推论。这是方法论正确的证据。
2. **基础设施是真的。** 智能体、执行沙箱、反事实构造、覆盖率校验、5 个模型家族的
   采样管线、22 个测试——这些可以整体迁移到任何一个新选题上，不需要重写。

**下一步应当先做的事：把 2607.01953、ClarifyGPT、Orchid、ClarifyCodeBench 四篇精读，
确认上表每一条重合判定，然后带着「c」或「d」重新开题。**
