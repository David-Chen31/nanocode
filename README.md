# Ask-or-Act

一个小型编程智能体，外加一套用来研究「智能体什么时候该向人提问」的实验装置。

智能体本身刻意做得很小——它是研究的**对照条件**。所有"聪明"的部分都放在 `askoract/`
里，做成可以整体关掉、可以被单独测量的策略模块。

```
agent/        编程智能体：工具调用循环、受限工作区、完整轨迹与成本记录
askoract/     研究部分：探针合成、差分执行、行为熵、提问生成、提问门控
bench/        诊断任务集（12 个反事实歧义任务）、oracle 用户模拟器、评测装置
experiments/  三个实验 + 图表
results/      实验产物（JSON、图、轨迹）
tests/        22 个单元与端到端测试
```

## 环境

Python 3.12+。实验部分只需要标准库；出图需要 matplotlib；接真实模型才需要
`anthropic` 或 `openai`。

```bash
pip install -r requirements.txt
```

本机验证用的解释器：`D:\Users\28170\AppData\Local\Programs\Python\Python3.12.1\python.exe`
（Git Bash 里的 `python` 指向 msys2 的 3.14，没有 pip，别用那个）。

## 三条命令

```bash
# 1. 跑智能体（无 API key 时走离线 fixture，会真的写文件、真的执行代码）
python -m agent.cli "Write remove_outliers(xs, k) in outliers.py ..." --workspace ./demo_ws

# 2. 跑全部实验（约 80 秒，全离线，确定性）
python experiments/run_experiments.py

# 3. 出图
python experiments/make_figure.py
```

接了真实模型之后还有：

```bash
python experiments/sample_live.py --model gpt-4o-mini --k 6   # 采样，约 $0.007
python experiments/run_experiments.py --candidates results/live/mini-naive.json --out results/exp_mini-naive.json
python experiments/compare_live.py                            # fixture vs live 对比表
python experiments/diagnose_live.py                           # 逐任务多样性与覆盖率
```

测试：`python -m pytest tests/ -q` → 22 passed

## 接真实模型

三个后端共用一个接口。默认走 fixture，所以整套东西没有 key 也能完整复现。

```bash
export ANTHROPIC_API_KEY=...
export AOA_BACKEND=anthropic:claude-sonnet-5
python -m agent.cli "..." --workspace ./ws
```

或 `AOA_BACKEND=openai:gpt-4o-mini`（认 `OPENAI_BASE_URL`，可指向兼容端点）。

## 研究部分在做什么

**问题。** 真实会话里用户需要纠正智能体的比例约 44%，而智能体主动澄清的比例只有 1–2%。
它宁可猜，也不问。什么时候该问，目前没有原则性的判据。

**做法。** 不用 token 级 logprob 度量不确定性——在代码上它分不清无害的语法差异和致命的
逻辑差异。改成：对同一个欠规范任务采样 k 份候选实现，用**差分执行**把它们聚成行为等价类，
在等价类上算熵（BSE），再用这个熵决定要不要打断用户、以及具体问哪一个输入。

**诊断集。** 12 个任务，每个有两份提示：完整版，和删掉恰好一句约束的版本。被删的那句就是
ground truth，所以不仅能测最终成功率，还能测「它有没有问对地方」。集合里故意放了 3 个
**困难负例**——样本会一致收敛到常见读法，而那个读法是错的。没有这类样例的基准会让任何基于
分歧的方法看起来都很好。

**两轮结果。**

第一轮（手写候选实现）：BSE 检测欠规范 AUROC **0.986**，词法基线 0.514。看起来很成功。

第二轮（真实模型采样，`gpt-4o-mini` / `gpt-4o`，720 次调用，$0.117）：**没有复现。**
BSE 掉到 0.389。原因可测——真实模型 6 份采样的平均行为类只有 **1.08**（手写的是 2.83），
温度提到 1.6 只到 1.17，主动诱导多样性只到 1.33。**朴素采样根本产生不了行为分歧。**
而且删掉一句规范后，正确行为只在 **6/12** 的候选集里出现过——给完整提示时是 12/12，
所以这不是基准的问题。

主结论：**「该不该问」所需的信息不在模型的输出分布里。** 分散度、logprob、自报置信度
全部失败；唯一有一点信号的是直接读需求文本（`direct_ask` 0.604，但置信区间含 0）。

详见 [docs/FINDINGS.md](docs/FINDINGS.md)。

## 重要限制

- **n = 12**，统计效力很弱。所有区间都很宽，**没有任何一个 AUROC 差异是显著的**。
  结论应读作「大效应没有复现」，而不是「某信号确定失败」。
- **只测了一个模型家族**（gpt-4o-mini / gpt-4o）。这是最该补的一步。
- **任务都是单函数级**，模型在这类常见函数上近乎确定性，可能不外推到仓库级。
- **用户模拟器就是参考实现**：永远在线、永远正确、永远不烦。
- 第一轮的 fixture 数值**不要当实证结论引用**——第二轮已证明它们是手写多样性的产物。
  保留它是因为它验证了测量机制本身正确，且离线可完整复现。
