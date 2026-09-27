# AGENTS.md — 关于持续学习、冷启动与苦涩的教训

本目录是 `mcts-theory-learning` 的增补数学文档集。修改前注意：

## 1. 事实基准

- AlphaZero/AlphaGo Zero/MuZero/MiniZero 的事实以本目录 `01` 章与
  `/mnt/gloway/projects/google-zero-series-data`（远端）为准：
  AGZ/AZ 是目标棋盘 tabula rasa、无小棋盘课程；MiniZero 的 progressive simulation
  是"预算课程"；不要写成"AlphaZero 先训练小棋盘"。
- AlphaProof 事实以 `alpha-proof-original-math-version` 档案 02/03/06/12 为准：
  价值 $d=-V$（期望剩余步数；AND max、OR min）、$Q=\gamma^{d-1}$、
  策略目标是 one-hot 预测实际 tactic、失败/超时被过滤。
- 既有结论引用不重复：mstc-rl 系列（删失 (Z4)(Z5)、指数墙、定理 A/B/C、方案 1–8）、
  Jev×AP 03（删失工具箱）、OPD 03/06（AWR/镜像下降、链式指数墙）。

## 2. 写作与编号

- 全局公式编号 (Z1)–(Z8) 在 `00` §3 登记；各章局部编号 (x.y) 自洽即可。
- 命题/定理：假设 → 陈述 → 证明（或"证明要点 + 缺口"）；反例必须可手动验证。
- 推测性内容标 `(S)`；每章末尾维护"(S) 与存疑"小节；引用经典结果不编造定理号与常数。

## 3. 渲染与校验（强制）

```bash
node ~/.config/opencode/skills/md-latex-rule/scripts/check_md_math.mjs [0-9][0-9]-*.md
```

要求 `OK` 或 `REAL REMAINING ISSUES: 0`。块级公式双美元标记、前后空行、独占行、零缩进；
行内公式不跨行、与 ASCII 字母数字之间留空格；公式内禁用 Unicode 数学符号。

## 4. 交叉引用

- 跨章引用格式："第 04 章命题 4.7"、"mstc-rl 04 式 (14)"、"档案 06 §2"。
- 修改编号后用 `rg` 全目录同步。
