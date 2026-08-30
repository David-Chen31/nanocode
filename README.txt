nanocode —— 从零手写的编程智能体
https://github.com/David-Chen31/nanocode

它与大模型对话，自主读写文件、搜索仓库、执行命令，直到完成或判断自己完不成。
未用任何 agent 框架或 SDK，依赖只有 openai / anthropic 两个厂商客户端库。
凭据仅从环境变量读取，仓库内无配置文件。

运行：
  pip install -r requirements.txt
  export OPENAI_API_KEY=...   NANOCODE_BACKEND=openai:claude-sonnet-5
  python -m agent.cli "任务描述" --workspace ./ws
离线 fixture 后端无需 key 即可复现；97 个测试确定性、不花钱。

题目要求手写的五项：对话历史与上下文管理 agent/context.py；工具定义与本地执行
tools.py + search.py + workspace.py；模型输出解析 llm.py；循环终止条件 loop.py；
错误处理散见以上各处。

四个设计决定，每个都有实测支撑：

一、上下文驱逐时改写消息内容，不删除消息。两种 wire format 都把工具结果与请求它的
assistant 消息配对，删掉一条得到的是畸形请求，而不是更短的请求。

二、命令输出按流分别解码。子进程写 UTF-8，命令不存在时却是 shell 用 OEM 代码页写
错误。钉死 UTF-8 会把「不是内部或外部命令」变成乱码——实测智能体因此在 14 步
里烧掉 10 步猜命令。

三、执行同时受时间与体积两个上限。实测 while True: print 三秒产出 181MB，30 秒超时
等于放行约 1.8GB，一个 432 条轨迹的实验就是这样 MemoryError 死掉的。

四、终止条件。曾有 12/18 条写对代码的轨迹从不调用 finish。顺数据查下去根因不在模型：
沙箱里 python 指向另一个解释器，测试根本跑不起来。修复后模型调用降低 22%。

我还对自己加的部件做了消融（144 条轨迹，预注册在先）：搜索工具对正确率的影响是
0.0 个百分点，却让每次运行贵近一倍。这个结果对我不利，但它是数据，我保留了它。
