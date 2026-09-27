# 关于持续学习、冷启动问题与苦涩的教训

`mcts-theory-learning` 增补系列：AlphaZero → AlphaProof → 真实世界经验学习。

- 目标：把"苦涩的教训是否意味着放弃预训练/纯 TTRL""AlphaZero 冷启动机制与 4×4 之问"
  "AlphaProof 与 AlphaZero 的可枚举/可变动作空间差异""如何扩展到具身与真实世界交互"
  写成公式级分析。
- 方法：子智能体按统一简报（关键公式 + 直觉 + 上下文）分章撰写，主智能体统一编号、
  交叉引用与校验；已写过的结论（mstc-rl 系列、mcts 笔记、AlphaProof 档案、Jev/OPD）
  直接引用不重复。
- 材料：`/mnt/gloway/projects/google-zero-series-data`（远端 Lexar/F 盘的 Zero 系列资料，
  含 AGZ/AZ/MuZero/MiniZero 论文与数学笔记）。

## 文件

- `00-导读与四问速答.md` — 四问答案、公式索引、质量说明
- `01-AlphaZero自我博弈与冷启动的事实核查.md`
- `02-苦涩的教训的形式化与纯TTRL之问.md`
- `03-冷启动的数学-有效分支因子与课程定理.md`
- `04-结构差异-可枚举动作空间与可变状态依赖动作空间.md`
- `05-真实世界反馈的数学-部分可观测与世界模型与预测编码.md`
- `06-具身与工具化的扩展-权限约束与硬件接口与持续学习.md`
- `07-结论-设计建议与开放问题.md`

## 约定

- `(S)` 表示推测性建议、未验证常量或模型内假设；每章末尾有存疑小节。
- 引用经典结果只作定性使用；公式编号 (Z1)–(Z8) 见 `00` §3。
- 公式渲染遵循 `md-latex-rule`：

```bash
node ~/.config/opencode/skills/md-latex-rule/scripts/check_md_math.mjs [0-9][0-9]-*.md
```

要求输出 `OK` 或 `REAL REMAINING ISSUES: 0`。
