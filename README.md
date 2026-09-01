# nanocode

一个从零手写的编程智能体：它和大模型对话，自己读写文件、搜索仓库、执行命令，
直到把任务做完或者判断自己做不完。没有用任何 agent 框架或 SDK。

```bash
python -m agent.cli "给 utils.py 里的每个函数补上 docstring" --workspace ./sandbox
```

**每个设计决定和支撑它的实测数据，压在一页里：[DECISIONS.md](DECISIONS.md)**
——包括证据推翻了我理由的三行，和至今没有证据的两行。

nanocode 的竞争点不是“又一个最小 ReAct 循环”。它要回答更难也更实用的问题：
**一个 agent 部件究竟值回了多少质量、token 和失败风险？** 因此实验会保存随机 schedule、
完整轨迹、最终 patch、运行环境和失败关闭的统计裁决；证据不支持作者时，首页保留被推翻的理由。
完整定位与路线图见 [docs/COMPETITIVE_STRATEGY.md](docs/COMPETITIVE_STRATEGY.md)。

## 它是怎么转的

一个 ReAct 循环：模型看到工具清单，选一个调用，拿到结果，再选下一个。
循环体在 [agent/loop.py](agent/loop.py)，一共 215 行，能一口气读完。

```
agent/
  loop.py       循环本体：终止条件、工具分发、错误恢复
  llm.py        三个后端（Anthropic / OpenAI 兼容 / 离线 fixture）共用一个接口
  tools.py      9 个工具的定义、JSON schema、参数校验、本地执行
  search.py     仓库搜索（正则搜内容 + glob 搜文件名）
  context.py    对话历史与上下文预算管理
  workspace.py  受限工作区：路径逃逸拦截、命令执行的时间与体积双重上限
  trace.py      支持保存模型调用与工具调用的完整记录，含 token 与成本
```

工具：`list_files` `read_file` `write_file` `edit_file` `search` `find_files`
`run` `ask_user` `finish`。

**终止条件是显式的三条**：调用了 `finish`；模型不再发起工具调用（把这一轮的文本当作
最终回答）；步数预算耗尽。提问预算与步数预算分开计，问完了会收到一条「自己判断并说明
假设」的回复——而不是沉默，沉默看起来跟工具坏了没区别。

## 五个值得辩护的设计决定

**① 驱逐上下文时改写消息，不删除消息。**
两种 wire format 都把工具结果和请求它的 assistant 消息配对（`tool_call_id` /
`tool_use_id`）。删掉一条得到的不是更短的请求，是**畸形的请求**，服务端整个拒掉。
所以回收内存的做法是把 content 换成一句「这里原本是 read_file pkg/helpers.py 的输出，
需要就重跑」。**有标签的空洞诱导模型重读，静默的空洞诱导它编造。**

实测：真正扛住负载的是「捕获时就截断」这条规则，不是驱逐——4 次被截断的读取把历史压在
2337 tokens / 2600 预算，驱逐一次没触发。这写在 [agent/context.py](agent/context.py)
的文档里，驱逐仍然保留：截断限制单条消息的开销，但没有任何东西限制消息的**数量**。

**② 搜索自己实现，而不是让模型敲 `grep`。**
它有 `run`，本来可以敲。但 Windows 上没有 grep 而模型不知道自己在哪个平台；shell 引号
会横在模型和它的意图之间；`grep -rn` 会把无界输出直接灌进上下文，顺便遍历 `.git`。
单文件上限与总上限**分开**：去掉单文件上限验证过，一个 100 匹配的文件会吃光总预算，
让另一个文件彻底不可见。截断永远带计数，因为智能体问「谁调用了这个函数」拿到 3 条，
就会认定有 3 个调用者。

**③ 命令输出按流分别解码。**
这两条管道有**两个写入方**：子进程被要求写 UTF-8，但命令不存在时子进程根本没启动，
是 shell 在写错误——用 OEM 代码页。钉死 UTF-8 会让 `'pytest' 不是内部或外部命令`
变成一行 `����`。实测代价：智能体读不懂这句话，烧掉 14 步里的 10 步猜命令写法。
修好后同一条轨迹 11 步完成。

**④ 执行同时受时间和体积两个上限。**
`capture_output=True` 用的是无界 `fh.read()`。智能体写出打印循环时超时**根本没机会触发**
——进程完全在照指令办事，只是永远不停。实测 `while True: print('x'*4096)` 三秒 181 MB。
一个 432 条轨迹的实验就是这样死的（MemoryError，已完成的全部丢失）。现在输出落到临时文件，
在同一个轮询里同时查时钟和查体积，触发时在 stderr 里说 "probably looping"。

**⑤ 模型犯错不终止 run。**
`json.loads(tc.function.arguments)` 原本没有保护——`content` 撞上 token 上限的
`write_file` 会产生截断 JSON，从后端内部抛出，位置和严重性都错。现在参数校验在分发**之前**
对着工具自己的 schema 做，消息用模型见过的名字，而不是 `<locals>.write_file() missing 1
required positional argument`。未知工具名会列出真实工具——模型伸手去够 `grep` 时能据此
改用 `search`，而「没有」这个回答它没法据以行动。**类型故意不校验**：把 `"3"` 转成 `3`
属于掩盖真实接口分歧的「帮忙」。

## 跑起来

Python 3.12+。核心只依赖 `openai` / `anthropic` 两个厂商客户端库（题目允许），
出图用 matplotlib，测试用 pytest。

```bash
pip install -r requirements.txt

# 离线 fixture 后端，不需要 key，会真的写文件、真的执行代码
python -m agent.cli "..." --workspace ./demo_ws

# 接真实模型
export OPENAI_API_KEY=...        # 凭据只从环境变量读，仓库里没有配置文件可泄漏
export NANOCODE_BACKEND=openai:gpt-4o-mini
python -m agent.cli "..." --workspace ./demo_ws
```

`NANOCODE_BACKEND` 也接受 `anthropic:claude-sonnet-5`；OpenAI 那支认
`OPENAI_BASE_URL`，可指向兼容端点。

测试：`python -m pytest tests/ -q` → **155 passed**，全部离线、确定性、不花钱。
其中大部分钉的是**会安静出错**的地方：路径逃逸、edit 锚点不唯一、工具结果配对不变量、
截断的 JSON、失控的打印循环。

本机验证用的解释器：`D:\Users\28170\AppData\Local\Programs\Python\Python3.12.1\python.exe`
（Git Bash 里的 `python` 指向 msys2 的 3.14，没有 pip）。

## 另外半个仓库：研究部分

`askoract/` `bench/` `experiments/` 是一套实验装置，用来回答一个具体问题：
**智能体什么时候该停下来问人，而不是猜。** 它不是这个作业的交付物，但它是上面那些
设计决定的来源——尤其是 `ask_user` 为什么长这样、为什么提问预算独立于步数预算。

真实会话里用户需要纠正智能体的比例约 44%，而智能体主动澄清只有 1–2%。它宁可猜。

做法：对同一个欠规范任务采样 k 份实现，用**差分执行**把它们聚成行为等价类，在等价类上
算熵。诊断集是 12 个任务，每个有两份提示，**只差恰好一句约束**——被删的那句就是
ground truth，所以不只能测最终成功率，还能测「它有没有问对地方」。

七轮结果，包括推翻前几轮的那些，记在 [docs/FINDINGS.md](docs/FINDINGS.md)。
一句话总结：**「该不该问」所需的信息不在模型的输出分布里**——分散度、logprob、自报置信度
全部失败；唯一上过随机的信号是让它去读仓库（ranking 0.82），但那个信号对**错误的**
约定是盲的。[docs/PREREG_*.md](docs/) 是实验前写下的预测，上半部分**至今未改**，
包括预测错了的那些。

确认性重跑的协议在 [docs/PREREG_confirmatory.md](docs/PREREG_confirmatory.md)：
条件在 task × repetition 块内随机，带补丁的消融逐轨迹进程隔离，结果保存运行 manifest、
完整 trace、最终 patch 和 held-out 评分。固定任务 estimand 使用 `analyse.py --fixed-tasks`，
不再把 12 个目的性任务靠 bootstrap 解释成一般任务总体。

为了单独检验外部效度，仓库新增了一层**开源时间留出数据**：从 6 个第三方 Python 项目中，
按提前提交的规则机械选出 26 个在固定模型 snapshot 之后创建并合并的真实 PR。每题冻结 base
SHA、PR、许可证、代码/测试文件和 patch SHA-256；gold 与隐藏测试保存在 agent 工作区之外。
选择规则和不足见 [docs/PREREG_open_source_data.md](docs/PREREG_open_source_data.md)，数据在
[bench/open_source_tasks.json](bench/open_source_tasks.json)。它当前是候选外部集，只有通过
base-red / gold-green 环境验证的题才会进入模型实验。

## 已知限制

- **搜索是最近才加的**，此前的轨迹实验里「智能体不去探索」这个结论因此有混淆变量：
  它当时没有搜索工具。那批数据与新代码**不可混用**，要重跑。
- `run` 是本地 subprocess，不是安全沙箱。工作区限制的是**波及范围**，不是权限。
- 研究部分 n = 12，统计效力弱，所有区间都很宽。结论应读作「大效应没有复现」。
- 用户模拟器就是参考实现：永远在线、永远正确、永远不烦。
