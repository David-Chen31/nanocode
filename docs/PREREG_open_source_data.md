# 开源时间留出集：抽样规则

状态：**在生成候选清单与运行任何 agent 之前写定。** 本文件只规定数据选择，不预注册模型效应。

## 目的

现有 `bench/tasks` 的 12 题是作者目的性构造的机制诊断集，`bench/repo_tasks.py` 的 8 题来自
nanocode 自身。它们有精确行为 ground truth，但不能代表外部开源项目。本数据层只回答：

> 在作者未参与维护的真实 Python 仓库、真实 issue/PR 和真实测试上，结论能否复现？

它不替代自制诊断集，两层结果不得混成一个总体正确率。

## 时间留出

- 冻结模型参照：`gpt-4o-mini-2024-07-18`。
- PR 合并时间窗：`2024-07-19T00:00:00Z` 至 `2025-07-18T23:59:59Z`。
- 因任务在 snapshot 之后合并，这个固定模型不可能在训练时见到最终 PR；这不保证其他模型
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

对每个仓库调用 GitHub Search API，查询上述时间窗内 merged PR，按创建时间升序，最多取前
100 个作为候选池。依次检查，取最先满足条件的 5 个：

- 作者不是 bot；标题不包含 release、dependency bump、pre-commit、文档/拼写专用变更；
- patch 同时修改至少一个非测试 `.py` 文件和至少一个独立测试文件；
- 总修改文件数 2–8；
- GitHub 报告 additions ≤ 300、deletions ≤ 200；
- PR 确实在时间窗内 merged，base/head/merge SHA 均存在；
- 仓库许可证 SPDX 在 `MIT`、`BSD-3-Clause`、`Apache-2.0`、`ISC` 中。

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
