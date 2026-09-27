# V1 MCTS 完整运行流程（reap-mcts-lean-v1）——逐轮拆解 + 手算示例

一句话：V1 的 MCTS 是**跑在 Lean 进程里的 CPU 搜索内核**（上游 reap 的 `Reap.Tactic.TreeSearch`），GPU 只在"展开一个叶子"时被调用一次；每轮迭代 = 下降（PUCT argmax）→ 叶子展开（一次策略/价值前向）→ 折扣回传，直到根节点被判定 solved 或预算耗尽。

## 1. 代码在哪（V1 全量归集）

CPU 侧（搜索内核，Lean）：

- 主引擎：`/mnt/gloway/projects/reap-new-update-model/new-v1-gather-source-code-cpu/reap-upstream/Tactic/TreeSearch.lean`
  - 主循环 `monteCarloTreeSearch`（:470）、单步 `reapMCTSStep`（:439）、PUCT `computePUCTScores`（:312）、progressive sampling `shouldProgressiveSample`（:333）、备份 `updateEdge/updateNode/backpropValueTowardsMin`（:386–412）。
- 通用模板（未被实际调用）：`reap-upstream/TreeSearch/MCTS.lean:86`。
- 超参数选项：`reap-upstream/Options.lean`。
- GPU 边界：`reap-upstream/Tactic/Generator.lean:149` `generatePolicyValue`。
- WSL 本地对照：`/home/zhai/project/reap/Reap/…`（其 `TreeSearch.lean` 与远端归集 md5 一致：`6c53253ce0d2d493da007215e4f047c1`）。

编排侧（Python，仅胶水；搜索/验证仍在 Lean）：

- `python-driver/v1_run.py`（BatchSolver：生成临时 `.lean` → `podman run … lake env lean` → 以 `%%TASK_<id>_DONE%%` 判完成；`--workers` 解析了但未实现）。
- `python-driver/v1_sink.py`、`python-driver/mock_policy_server.py`。

GPU 侧：

- `v1-spec/v1-1-training-example/scripts/policy_server.py`（Flask :8000，`/v1/chat/completions` + `/value`）、`ps_server.py`（:8081，`/premises`）。
- `app/` 下另有 RTTT/TTTRT 版 `policy_server.py`、`value_head.py` 等。

调用链：`v1_run.py` → Lean `reapMCTS`（`Tactic/Syntax.lean:233`）→ `runMCTS` → `monteCarloTreeSearch` → `reapMCTSStep` → 叶子 `visitNode` → `evalPolicyValue = generatePolicyValue` → 三个 HTTP 端点。

## 2. 每轮迭代做什么

```
while 节点数 <= maxNodes && step < maxSteps:      # 默认 64 / 64
    reapMCTSStep(root)                            # 一次 simulation
    if root.isSolved: return 找到解
    step += 1
```

单步 `reapMCTSStep(i)`：

```
if children 为空 或 shouldProgressiveSample(node):
    visitNode(i)                                  # 展开：调 GPU，把成功 tactic 挂成孩子
    return 该节点的价值
else:
    k = argmax(Q + U)                             # PUCT（纯 CPU，严格大于、平局取先出现）
    childValue = reapMCTSStep(孩子 k)
    value = childValue - stepCost(边 k)           # 普通 tactic 边=1，focus 边=0
    edge.numVisit++, node.numVisit++, node.valueSum += value
    return 回传值                                 # OR 原样；AND 取未解孩子的 min
```

要修正的直觉：

- 不是"先采样再选择"。**每轮先 PUCT 选择下降，走到叶子才调 GPU 一次**；第 1 轮根没有孩子，直接展开，所以**第一次 GPU 调用发生在根**，此时还谈不上选择。
- GPU 只对当前叶子做一次前向，**不会沿树"往下 roll"**；没有 playout/rollout，叶值=价值头直接输出。
- 一次展开 = 一次 `evalPolicyValue`：premise（16 条）+ policy（n=6 个候选，带 logprob）+ value（1 个分数），其中 policy 与 value 并发（`CoreM.asTask`）。选择/备份不碰 GPU。
- 终止：根 `isSolved`（子目标全解）或 `maxNodes`/`maxSteps` 耗尽；解出的脚本最后还要 `replaySolvedNode` + `checkProof` 内核复验才算真成功。

## 3. 关键公式

探索系数（随节点访问量对数增长）：

$$
c(N) = c_{\mathrm{init}} + \ln\!\left(\frac{N + c_{\mathrm{base}} + 1}{c_{\mathrm{base}}}\right),
\qquad c_{\mathrm{init}} = 0.001,\ c_{\mathrm{base}} = 3200 .
$$

选择索引（PUCT，argmax 决策）：

$$
\mathrm{score}(s,a) = Q(s,a) + c(N)\,\frac{p_a}{\sum_b p_b}\,\frac{\sqrt{N}}{N(s,a)+1},
$$

其中先验来自 LLM 的 token logprob 之和再经温度缩放：$\text{probability} = \exp(\text{logprob}/\tau)$，$\tau = 50$（`TreeSearch.lean:276`）。

价值变换（值头输出统一先取负，$v$ 越小越"难"）：

$$
\text{valueScore} = \gamma^{-1-v},\qquad v = \text{child.value} - \text{stepCost},
\qquad \gamma = 0.99 .
$$

- OR 节点：$Q = \text{valueScore}$（越大越好）。
- AND 节点：$Q = 1 - \text{valueScore}$；已解孩子 $Q = -\infty$（不再选它）。
- 回传：OR 原样通过；AND 取"未解且有访问"孩子的最小值（min 从 1.0 起）。
- 终局：解出的孩子 $\text{valueScore} = \gamma^{0} = 1$；每多一步就多远一个 $\gamma$，等价于"短证明优先"。

## 4. 手算示例（a1–a4）

场景：根 `s0 : P ∧ Q`，上下文有 `hP : P`、`hQ : Q`。默认超参：$\gamma = 0.99$，$c_{\mathrm{init}} = 0.001$，$c_{\mathrm{base}} = 3200$，$\tau = 50$，$C = 0.01$，$\alpha = 0.6$，`num_samples = 6`，`maxNodes = maxSteps = 64`。数字均为随意取的演示值。

### 迭代 1：根展开（GPU 调用 #1）

根没有孩子 → `selectChild` 返回 `none` → `visitNode(根)`。一次 GPU 轮：

- value 头：`score = 4`（当作"还差 4 步"）→ 代码取负存入节点：根 `valueSum = -4`、`numVisit = 1`、`numEvaluations = 1`。
- 策略采样 4 个候选（演示只用 4 个）：

| 动作 | 内容 | 先验 |
|---|---|---|
| a1 | `constructor` | $p_1 = 0.40$ |
| a2 | `left` | $p_2 = 0.30$ |
| a3 | `simp`（执行失败） | $p_3 = 0.10$ |
| a4 | `by_contra h` | $p_4 = 0.10$ |

逐个在 Lean 里执行：

- a1 成功：目标裂成 `g0 : P`、`g1 : Q`，且两个目标都不含未赋值 mvar → 子节点 **s1 是 AND 节点**。AND 节点创建时立刻挂 **focus 孩子**：`focus_goal 0`（f0）、`focus_goal 1`（f1），先验各 $1/2$，边长代价 0，边价值 0。
- a2 成功 → OR 孩子 s2（目标 `P`，未解）。
- a3 失败 → 丢弃（树上没有它）。
- a4 成功 → OR 孩子 s3（未解）。

根有 3 个孩子（a1、a2、a4），都 `numVisit = 0`。没有任何孩子 solved，继续。

### 迭代 2：选择根 → 选择 AND 子目标 → 展开 f0（GPU 调用 #2）

根 PUCT（$N = 1$，只有 a3 不在树上，先验重新归一化到 $0.80$）：

$$
c = 0.001 + \ln\!\left(\frac{3202}{3200}\right) \approx 0.001625 .
$$

| 孩子 | 归一化先验 $p$ | $Q$（未访问→0） | $U = c\,p\,\sqrt{1}/(0+1)$ | score |
|---|---|---|---|---|
| a1→s1 | 0.500 | 0 | 0.00081 | 0.00081 |
| a2→s2 | 0.375 | 0 | 0.00061 | 0.00061 |
| a4→s3 | 0.125 | 0 | 0.00020 | 0.00020 |

→ 选 **a1**，下到 s1（AND）。

s1 PUCT（$N = 0$，$\sqrt{N} = 0$）：两个 focus 孩子都没访问过，$\text{valueScore} = 0$，$Q_{\mathrm{AND}} = 1 - 0 = 1$，都并列 → 取先出现的 **f0**。

f0 没有孩子 → 展开，GPU 调用 #2（目标 `P`）：

- value 头：`score = 3` → $p' = -3$；f0：`valueSum = -3`、`numVisit = 1`。
- 策略：`b1 = exact hP`（$p = 0.6$）成功，直接把 `P` 关掉 → OR 孩子 t1（`isSolved = true`）；`b2 = simp`（$p = 0.4$）成功 → OR 孩子 t2（未解）。
- f0 因"有孩子 solved"而 solved；s1（AND）还需要 f1，根未解。

回传：

- s1 处：$v = -3 - 0 = -3$（focus 边免费）→ s1 `numVisit = 1`、`valueSum = -3`。
- `backupValueForParent(s1)`：AND 只在"未 solved 且有访问"的孩子里取 min；f0 已 solved、f1 未访问，都不计 → 从 1.0 起 → 返回 **1.0**。
- 根处：$v = 1.0 - 1 = 0$ → 根 `numVisit = 2`、`valueSum = -4 + 0 = -4`。根仍未解。

### 迭代 3：根再选 a1 → AND 改攻另一个子目标 → 收束

根 PUCT（$N = 2$，$c \approx 0.001937$，$\sqrt{2} \approx 1.414$）：a1 的孩子 s1 已有价值 $v_{s1} = -3$，于是

$$
Q(\text{a1}) = \gamma^{-1-(-3-1)} = \gamma^{3} \approx 0.9703,
\qquad
U(\text{a1}) \approx 0.001937 \times 0.5 \times 1.414 \approx 0.0014 .
$$

a2、a4 仍是 $Q = 0$、$U \approx 0.001 \cdot p$ → 仍选 **a1**。

s1 PUCT（$N = 1$）：

| focus 孩子 | $v$ | valueScore | $Q$ | 结果 |
|---|---|---|---|---|
| f0（已解） | -3 | $\gamma^{2} \approx 0.9801$ | $1 - 0.9801 \approx 0.0199$ | 低 |
| f1（未访问） | — | 0 | $1 - 0 = 1$ | 高 → 选它 |

→ 改攻 **f1**（这就是 AND 节点"专攻当前未解瓶颈"的语义）。

展开 f1（目标 `Q`），GPU 调用 #3：value `score = 2` → $p' = -2$；策略 `d1 = exact hQ`（$p = 0.7$）成功关掉 `Q` → 孩子 t3 solved。

回传：

- s1 处：$v = -2 - 0 = -2$ → s1 `numVisit = 2`、`valueSum = -5`，`isSolved = 所有 focus 孩子已解 = true`。
- `backupValueForParent(s1)`：已无未解孩子 → 返回 **1.0**。
- 根处：$v = 1.0 - 1 = 0$；根的 `isSolved = 有孩子(s1)已解 = true`。

主循环发现根 solved → 返回解；`replaySolvedNode` 重放脚本、`checkProof` 内核复验，输出：

```
constructor
· exact hP
· exact hQ
```

一句话看懂数字：`Q = γ^{−1−v}`，越接近解越接近 1（1 步解出 = 1）；根选孩子看 $Q + U$，但默认 $c$ 极小（约 0.0016–0.002），所以**早期几乎纯按 $Q$（价值排序）选择，未访问孩子靠先验解锁**；到 $N \approx 15000$ 时 $c$ 才升到约 1.7。

## 5. 初始值有随机项吗？——没有

- 节点初值：`valueSum = 0`、`numVisit = 0`、`numEvaluations = 0`；边初值：`value = 0`、`numVisit = 0`。
- **未访问孩子的 $\text{valueScore}$ 直接规定为 0.0**（不是论文里的"父节点值 − 惩罚"，V1 没有 FPU）。
- 代码里搜不到 `dirichlet` / `noise` / `virtual`：没有 root Dirichlet、没有 root 温度采样、没有 virtual loss；选择是确定性 argmax。
- 唯一的随机性在**策略候选的生成**：LLM 采样 `temperature = 0.99`、`n = 6`（服务端；CPU mock 也是 random）。也就是说：随机性在"提出哪些 tactic"，不在"树统计量的初始化"。
- 对照：AlphaProof 论文本身没有 root Dirichlet；(S) 级别的建议见归档 `alpha-proof-original-math-version.wsl-backup/14-puct-and-temperature.md` §5–6、`17-design-choices.md` §G（root Dirichlet $\varepsilon, \alpha$、虚拟损失），V1 均未实现。另外论文对未访问边取 $V(s,a) = \hat V_{\mathrm{net}}(s) - c_{\mathrm{pen}}$，V1 代码直接用 0。

## 6. Progressive sampling：有代码，但默认几乎不触发

`shouldProgressiveSample`（`TreeSearch.lean:333`）：

$$
\text{命中} \iff \text{node 是 OR} \;\wedge\; n_{\mathrm{eval}} \le C \cdot N^{\alpha},
\qquad C = 0.01,\ \alpha = 0.6 .
$$

命中时 `selectChild` 返回 `none` → **不选已有孩子，而是在同一节点再调一次策略网络**，把新采样的 tactic 并入（去重合并先验），`numEvaluations += 1`。

实际行为（默认参数）：

- 节点第一次展开后 $n_{\mathrm{eval}} = 1$，而 $C \cdot N^{0.6}$ 在 $N \approx 2154$ 时才到 1；64 步预算内几乎永远不满足 $1 \le 0.01\,N^{0.6}$。
- 所以默认配置下它近似只起"先展开、后下降"的作用（展开全新的 $n_{\mathrm{eval}} = 0$ 节点）；想让它在关键路径上真正**加宽**，把 $C$ 调大（论文值 $C = 1.0$、$\alpha = 0.5$，且论文是"额外采 K 个动作"的 progressive widening）。
- AND 节点不参与该判断。注意 repo 里 `explain/10-mcts-usage-and-alternatives.md:72` 把它叫 "progressive widening"，代码里只有这一个机制。

## 7. 超参数（`Options.lean` 默认值）

| 选项 | 默认 | 生效值 | 含义 |
|---|---|---|---|
| `reap.max_goals` | 64 | 64 | 树节点上限 maxNodes |
| `reap.max_steps` | 64 | 64 | simulation 轮数上限 maxSteps |
| `reap.num_samples` | 6 | 6 | 每次展开采样的 tactic 数 |
| `reap.num_premises` | 16 | 16 | premise 检索条数 |
| `reap.c_base` | 3200 | 3200 | $c$ 的底数 |
| `reap.c_init` | 1 | 0.001 | $c$ 的初值（除以 1000） |
| `reap.visit_discount` | 990 | 0.99 | 折扣 $\gamma$（除以 1000） |
| `reap.prior_temperature` | 50 | 50 | 先验温度 $\tau$ |
| `reap.progressive_sampling_c` | 10 | 0.01 | 渐进采样 $C$（除以 1000） |
| `reap.progressive_sampling_alpha` | 600 | 0.6 | 渐进采样 $\alpha$（除以 1000） |
| `reap.temperature` | 99 | 0.99 | LLM 采样温度（除以 100） |
| `reap.max_tokens` | 1024 | 1024 | 单次生成上限 |
| `reap.timeout` / `reap.heartbeats` | 200000 / 1e9 | — | 单 tactic 超时 / 心跳 |

注意：`v1-spec/02-mcts-verifier.md` 写 `c_base = 3.2`，与代码冲突；以代码 `3200` 为准（否则 $c$ 会大 1000 倍）。

## 8. GPU 协议与已知坑

一次展开的三个 HTTP 请求：

1. `POST /premises`（或 premise 服务）：`{query, num_results=16}` → `[{formal_name, formal_statement}]`。
2. `POST /v1/chat/completions`：`{model, messages, n=6, temperature=0.99, max_tokens=1024, logprobs=true}` → OpenAI chat 格式；取 `message.content`（tactic 字符串）与 `logprobs.content[].logprob` 求和 = 该候选的 log 先验。
3. `POST /value`：`{state}` → `{"score": …}`；Lean 端对其取负存入节点。

坑：

- `python-driver/mock_policy_server.py` 返回的是 `text` / `logprob_avg`，与 Lean 客户端 `FromJson` 期望的 `message.content` / `logprobs` 不兼容；且 value 请求会落到 `/chat/completions` 后缀分支——本地端到端前需先修。
- 无跨节点批处理；批只体现在单次 policy 请求的 `n=6` 并行。一个后端实例同时只服务一个搜索。

## 9. 与论文的差异速查

- 未访问边初始化：论文 $V(s,a) = \hat V_{\mathrm{net}}(s) - c_{\mathrm{pen}}$；V1 用 0。
- Progressive sampling 参数：论文 $C = 1.0$、$\alpha = 0.5$（额外采 K 个 tactic）；V1 默认 $C = 0.01$、$\alpha = 0.6$（重采样验证时倾向于不触发）。
- Root Dirichlet / virtual loss：论文没有（属复盘建议 S）；V1 也没实现。
- 值头输出先取负、Q 用 $\gamma^{-1-v}$ 是工程侧约定；与论文的 return/步数记法等价换算见 [[8-6-q-transform-reason-1]]。

## 10. 参考

- 上游逐函数解读：`reap-source-code-explain/06-mcts-core.md`（PUCT/渐进采样/备份），`05-policy-value-premise.md`（选项与接口）。
- 论文对照：`alpha-proof-original-math-version.wsl-backup/03-search.md`、`13-mcts-theory.md`、`14-puct-and-temperature.md`。
- 本机副本：`/home/zhai/project/reap/Reap/Tactic/TreeSearch.lean`（与远端归集 md5 相同，行号可直接对照）。
- 关联笔记：[[6-alpha-proof-mcts-rl]]、[[8-6-q-transform-reason-1]]、[[9-mainrl-ttrl-llm-value-head]]。
