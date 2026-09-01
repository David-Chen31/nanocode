# 开源时间留出集：抽样规则

状态：**在生成候选清单与运行任何 agent 之前写定。** 本文件只规定数据选择，不预注册模型效应。

## 目的

现有 `bench/tasks` 的 12 题是作者目的性构造的机制诊断集，`bench/repo_tasks.py` 的 8 题来自
nanocode 自身。它们有精确行为 ground truth，但不能代表外部开源项目。本数据层只回答：

> 在作者未参与维护的真实 Python 仓库、真实 issue/PR 和真实测试上，结论能否复现？

它不替代自制诊断集，两层结果不得混成一个总体正确率。

## 时间留出

- 冻结模型参照：`gpt-4o-mini-2024-07-18`。
- PR 创建和合并时间都必须在：`2024-07-19T00:00:00Z` 至 `2025-07-18T23:59:59Z`。
- 因问题与最终 PR 都在 snapshot 之后公开，这个固定模型不可能在训练时见到它们；这不保证其他模型
  无污染，所以跨模型实验必须分别声明时间关系。

## 抽样框

按以下顺序固定六个第三方 Python 仓库，每仓库选 5 题，共 30 题：

1. `pallets/click`
2. `encode/httpx`
3. `pytest-dev/pytest`
4. `pydantic/pydantic`
5. `psf/requests`
6. `Textualize/rich`

选择依据只包括：Python 项目、公开 GitHub PR、持续维护、具备自动测试、宽松开源许可证；不依据
nanocode 或任何模型在其上的表现。

## 机械选择规则

对每个仓库调用 GitHub Search API，查询上述时间窗内创建且 merged 的 PR，按创建时间升序，最多取前
100 个作为候选池。依次检查，取最先满足条件的 5 个：

- 作者不是 bot；标题不包含 release、dependency bump、pre-commit、文档/拼写专用变更；
- patch 同时修改至少一个非测试 `.py` 文件和至少一个独立测试文件；
- 总修改文件数 2–8；
- GitHub 报告 additions ≤ 300、deletions ≤ 200；
- PR 确实在时间窗内 merged，base/head/merge SHA 均存在；
- 仓库许可证 SPDX 在 `MIT`、`BSD-3-Clause`、`Apache-2.0`、`ISC` 中。

### 生成前修订记录

第一次选择器运行在写出 manifest 前被人工中止：进度输出显示，仅限制 `merged_at` 会纳入
snapshot 前已经公开、但后来才合并的长期 PR。这与“模型不可能见过题目”的设计目的冲突。
因此在没有生成数据文件、没有运行 agent、没有查看任何模型结果时，增加 `created_at` 同样晚于
snapshot 的条件，并给 schema 加硬校验。Git 历史保留原规则和本修订。

若某仓库前 100 个候选不足 5 题，保留实际数量并显式报告，不向后扩大时间窗、不替换仓库。

## 冻结内容

`bench/open_source_tasks.json` 对每题保存：仓库、PR URL/编号、原始标题与正文、创建/合并时间、
base/head/merge SHA、许可证、修改规模、代码文件、测试文件、完整 patch URL 与 SHA-256。
manifest 还保存查询字符串、候选池大小、排除原因计数和生成器版本。

`patch_sha256` 让后续下载能发现上游内容漂移；base SHA 让仓库状态不随默认分支变化。

## 评分边界

- agent 只看到 base SHA 的仓库与 PR 标题/正文，不看到 gold patch。
- agent 停止后才把 gold patch 中的测试文件变更应用到 grader 副本。
- 行为测试与原回归测试都通过才算正确；基础设施/依赖安装失败单列。
- 在每题基线可复现、测试确实先红后绿之前，这 30 题只能叫“候选外部集”，不能报告模型分数。

## 复现命令

```bash
py -3 experiments/open_source_data.py select --out bench/open_source_tasks.json
py -3 experiments/open_source_data.py validate bench/open_source_tasks.json
```

重新运行 `select` 只用于审计选择过程；已冻结 manifest 不因 GitHub 后续变化自动覆写。

## 生成结果（选择后记录，不改规则）

选择器生成 **26 个候选任务 / 6 个仓库**：Click、HTTPX、pytest、Pydantic、Rich 各 5 个，
Requests 只有 1 个满足规则。按预注册没有替换仓库或扩大时间窗。

```text
tasks_sha256 = dfb9238b321ec68049ae28730a39bcd9842225825bad2c282b9aed0869f9b7b2
earliest created_at = 2024-07-22T11:57:36Z
latest merged_at    = 2025-06-05T14:55:33Z
```

这 26 个仍是候选集。物化冒烟发现第一题的新增测试在 base 上已经全过，证明“PR 修改了测试”
不等于“测试能区分修复”。因此 `validate-all` 会机械执行 base-red / gold-green 检查，并把
`BASELINE_ALREADY_PASSES`、`GOLD_DOES_NOT_PASS` 和 `INFRASTRUCTURE_ERROR` 全部保留，只有
`VALIDATED` 子集能进入模型实验。

物化与验证命令：

```bash
py -3 experiments/open_source_data.py materialize TASK_ID \
  --workspace results/_oss_workspace --grader results/_oss_grader
py -3 experiments/open_source_data.py validate-all \
  --out bench/open_source_validation_host_v2.json
```

## 验证审计（选择后记录，不改规则）

第一次主机审计暴露了两个物化错误：多提交 `.patch` 中同一路径会出现多次；测试进程也可能优先
导入主机已安装的同名包。完整 v1 记录保留在 `bench/open_source_validation_host_v1.json`。
修正方式不改变候选、顺序或纳入规则：base/head 均直接按冻结 SHA 下载，隐藏测试取 head 树中的
最终文件；测试时把对应工作树的 `src/` 和根目录置于 `PYTHONPATH` 最前。

v2 在 Windows、Python 3.12 主机上的结果为：

```text
VALIDATED                 7
BASELINE_ALREADY_PASSES   3
GOLD_DOES_NOT_PASS         1
INFRASTRUCTURE_ERROR      15
```

7 个有效任务来自 Click（4）和 Rich（3）。15 个环境错误主要来自 HTTPX 的 Trio 依赖、pytest 的
构建期版本文件和 Pydantic/Core 版本配对；在统一冻结环境重验前，不计入成功或失败。任何模型实验
必须用 `load_validated_open_source_tasks` 读取 v2，不能直接把 26 个候选当作评分分母。
