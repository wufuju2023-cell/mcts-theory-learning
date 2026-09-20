# 13 梯度下降、随机优化与自适应优化器

参考：本系列《10-grad-backprop-optimizer-gpu.md》§5（AlphaProof 的官方优化器事实与更新公式）、《11-probability-mle-estimation.md》（估计论铺垫）、《14-backprop-and-autodiff.md》（梯度怎么算出来）。本机旧笔记 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/06-main-rl.md` §4 有历史猜测（AdamW、$3\times10^{-5}$），以官方伪代码为准。

定位：把“最小化损失”的优化算法从凸情形的收敛证明一直推到 Adam/AdamW 的精确公式，并给出 AlphaProof 实际使用（`optax.adam`）与未公开部分的重建方案。

---

## 1. 优化问题的基本语言

### 1.1 问题与最优性

$$
\min_{\theta\in\mathbb{R}^{d}}\ f(\theta).
$$

- 若 $f$ 可微，$\nabla f(\theta^\star)=0$ 是局部最优的必要条件；
- 若 $f$ 凸，则该条件也是全局最优的充分条件。

### 1.2 凸性

$f$ 凸，当且仅当对任意 $\theta,\theta'$ 与 $t\in[0,1]$：

$$
f\big(t\theta+(1-t)\theta'\big)\le t f(\theta)+(1-t)f(\theta').
$$

可微函数的等价刻画：

$$
f(\theta')\ge f(\theta)+\langle\nabla f(\theta),\ \theta'-\theta\rangle
\quad\text{（一阶下界）}.
$$

二阶可微时：$\nabla^2 f(\theta)\succeq0$ 对所有 $\theta$。

### 1.3 光滑性与强凸性

- **$L$-光滑**（梯度 Lipschitz）：$\|\nabla f(\theta)-\nabla f(\theta')\|\le L\|\theta-\theta'\|$，等价于 $\nabla^2 f\preceq LI$；
- **$\mu$-强凸**：$f(\theta')\ge f(\theta)+\langle\nabla f(\theta),\theta'-\theta\rangle+\frac{\mu}{2}\|\theta'-\theta\|^2$，等价于 $\nabla^2 f\succeq\mu I$；
- **条件数** $\kappa=L/\mu\ \ge1$。它决定一阶方法的收敛速度。

---

## 2. 梯度下降

### 2.1 算法

$$
\theta_{k}=\theta_{k-1}-\eta\nabla f(\theta_{k-1}).
$$

### 2.2 下降引理（光滑性的直接推论）

**引理.** 若 $f$ 是 $L$-光滑，则对任意 $\theta,\theta'$：

$$
f(\theta')\le f(\theta)+\langle\nabla f(\theta),\theta'-\theta\rangle+\frac{L}{2}\|\theta'-\theta\|^2 .
$$

**证明（$C^2$ 情形）.** 由微积分基本定理与 $\|\nabla f(x)-\nabla f(\theta)\|\le L\|x-\theta\|$：

$$
f(\theta')-f(\theta)
=\int_0^1\langle\nabla f(\theta+t(\theta'-\theta)),\ \theta'-\theta\rangle\,\mathrm{d}t
$$

$$
=\langle\nabla f(\theta),\theta'-\theta\rangle
+\int_0^1\big\langle\nabla f(\theta+t(\theta'-\theta))-\nabla f(\theta),\ \theta'-\theta\big\rangle\,\mathrm{d}t
$$

$$
\le\langle\nabla f(\theta),\theta'-\theta\rangle
+\int_0^1 Lt\|\theta'-\theta\|^2\,\mathrm{d}t
=\langle\nabla f(\theta),\theta'-\theta\rangle+\frac{L}{2}\|\theta'-\theta\|^2 .
$$

$\square$

取 $\theta'=\theta-\eta\nabla f(\theta)$：

$$
f(\theta')\le f(\theta)-\eta\Big(1-\frac{L\eta}{2}\Big)\|\nabla f(\theta)\|^2 .
$$

所以当 $0\lt\eta\le1/L$ 时每步至少按 $-\eta(1-L\eta/2)\|\nabla f\|^2$ 下降；$\eta=1/L$ 给出 $f(\theta')\le f(\theta)-\frac{1}{2L}\|\nabla f(\theta)\|^2$。

### 2.3 凸情形的收敛率

**定理.** $f$ 凸且 $L$-光滑，取 $\eta=1/L$，则

$$
f(\theta_k)-f^\star\ \le\ \frac{L\|\theta_0-\theta^\star\|^2}{2k}.
$$

**证明要点.** 记 $r_k=\|\theta_k-\theta^\star\|$。由下降引理与凸性的一阶下界 $f(\theta_k)-f^\star\le\langle\nabla f(\theta_k),\theta_k-\theta^\star\rangle$：

$$
f(\theta_{k+1})\le f(\theta_k)-\frac{1}{2L}\|\nabla f(\theta_k)\|^2,
$$

$$
r_{k+1}^2=r_k^2-2\eta\langle\nabla f(\theta_k),\theta_k-\theta^\star\rangle+\eta^2\|\nabla f(\theta_k)\|^2
\le r_k^2-\frac{1}{L}\big(f(\theta_k)-f^\star\big)
$$

（最后一步用了 $\eta=1/L$ 与 $2\langle\nabla f,\theta-\theta^\star\rangle\ge2(f-f^\star)$ 及 $-\frac{1}{L}\|\nabla f\|^2$ 的抵消）。对 $k$ 求和并用 $r_k\ge0$：

$$
\sum_{i=0}^{k-1}\big(f(\theta_i)-f^\star\big)\le\frac{L}{2}\big(r_0^2-r_k^2\big)\le\frac{L}{2}r_0^2 .
$$

又 $f(\theta_k)$ 单调不增，故 $k\big(f(\theta_k)-f^\star\big)\le\sum_{i\lt k}(f(\theta_i)-f^\star)$，即得。$\square$

**强凸情形.** 若还有 $\mu$-强凸，则可证线性收敛：

$$
f(\theta_k)-f^\star\ \le\ \Big(1-\frac{\mu}{L}\Big)^{k}\big(f(\theta_0)-f^\star\big),
$$

即达到 $\varepsilon$ 精度需要 $O(\kappa\log(1/\varepsilon))$ 步。条件数 $\kappa$ 越大，等高线越“狭长”，梯度下降越慢——这是动量与预条件方法存在的理由。

---

## 3. 随机梯度下降（SGD）

### 3.1 随机梯度是无偏估计

设 $f(\theta)=\mathbb{E}_{\xi}[F(\theta;\xi)]$，其中 $\xi$ 是样本/小批量。用 $g(\theta;\xi)=\nabla_\theta F(\theta;\xi)$ 作为梯度估计：

$$
\mathbb{E}[g(\theta;\xi)]=\nabla f(\theta),
\qquad
\mathrm{Var}\big(g(\theta;\xi)\big)=\sigma^2(\theta).
$$

小批量 $B$ 条样本的平均估计方差缩小 $B$ 倍：

$$
\mathrm{Var}\Big(\frac{1}{B}\sum_{i\in B}g_i\Big)=\frac{\sigma^2}{B}.
$$

### 3.2 收敛率（叙述与要点）

凸、$L$-光滑、梯度方差有界 $\mathbb{E}\|g-\nabla f\|^2\le\sigma^2$ 时，固定步长 $\eta\le1/L$ 的 SGD 满足

$$
\mathbb{E}\big[f(\bar\theta_K)\big]-f^\star
\ \le\
\frac{\|\theta_0-\theta^\star\|^2}{2\eta K}
+\frac{\eta\sigma^2}{2},
$$

（$\bar\theta_K$ 是迭代平均）。第一项是优化误差，随 $K$ 下降；第二项是随机噪声地板，随 $\eta$ 缩小。若 $\eta_k=O(1/\sqrt{k})$，可得 $O(1/\sqrt{K})$；强凸下用 $\eta_k=O(1/(\mu k))$ 可得 $O(1/K)$。

**与 AlphaProof 的差异.** 论文的 learner 是“固定 batch 的确定性优化 + 数据不断刷新”，batch 4096、约 $10^{6}$ 步；搜索数据的分布随策略变化，因此严格说这是一个**非平稳随机优化**问题，收敛率定理只能作为直觉参考。

---

## 4. 动量方法

### 4.1 Heavy-ball 与 EMA

$$
v_k=\beta v_{k-1}+\nabla f(\theta_{k-1}),
\qquad
\theta_k=\theta_{k-1}-\eta v_k,
\qquad \beta\in[0,1).
$$

展开：$v_k=\sum_{i=1}^{k}\beta^{k-i}\nabla f(\theta_{i-1})$，是梯度的指数滑动平均（EMA）。作用有两个：

- 沿长期一致的方向放大步长（“加速”），在狭长峡谷中减小震荡；
- 平滑随机梯度的噪声（方差缩小约 $(1-\beta)/(1+\beta)$ 倍）。

Nesterov 动量把梯度取在“前瞻点”$\theta_{k-1}-\eta\beta v_{k-1}$ 上，凸光滑下有最优的 $O(1/k^2)$ 加速率（本系列不做完整证明）。

---

## 5. 自适应学习率：AdaGrad、RMSProp、Adam、AdamW

### 5.1 AdaGrad 与 RMSProp

AdaGrad 逐坐标累计梯度平方并缩放：

$$
v_{k,i}=\sum_{t=1}^{k}g_{t,i}^2,
\qquad
\theta_{k,i}=\theta_{k-1,i}-\eta\frac{g_{k,i}}{\sqrt{v_{k,i}}+\varepsilon}.
$$

效果是“梯度大的坐标自动减速”；问题是 $v$ 单调增长导致学习率终将消失，不适合长期训练。RMSProp 改成指数遗忘：

$$
v_{k,i}=\rho v_{k-1,i}+(1-\rho)g_{k,i}^2 .
$$

### 5.2 Adam 的精确公式

一阶矩与二阶矩（逐坐标）：

$$
m_k=\beta_1 m_{k-1}+(1-\beta_1)g_k,
\qquad
v_k=\beta_2 v_{k-1}+(1-\beta_2)g_k\odot g_k,
\qquad m_0=v_0=0 .
$$

**偏差修正.** 展开 $m_k$：

$$
m_k=(1-\beta_1)\sum_{i=1}^{k}\beta_1^{k-i}g_i
\quad\Longrightarrow\quad
\mathbb{E}[m_k]=(1-\beta_1^{k})\,\mathbb{E}[g]
$$

（设梯度均值平稳）。同理 $\mathbb{E}[v_k]=(1-\beta_2^{k})\mathbb{E}[g^2]$。因此

$$
\widehat m_k=\frac{m_k}{1-\beta_1^{k}},
\qquad
\widehat v_k=\frac{v_k}{1-\beta_2^{k}}
$$

是无偏矩估计，更新为

$$
\boxed{\;
\theta_k=\theta_{k-1}-\eta_k\,\frac{\widehat m_k}{\sqrt{\widehat v_k}+\varepsilon}.
\;}
$$

Optax 的 `optax.adam(learning_rate)` 默认 $\beta_1=0.9$、$\beta_2=0.999$、$\varepsilon=10^{-8}$（`scale_by_adam` 内部实现上式，`scale_by_learning_rate` 默认取负号），与官方伪代码的 `self.optimizer = optax.adam(config.lr)` 对应。

### 5.3 数学解读

改写为对角预条件：

$$
\theta_k=\theta_{k-1}-\eta_k\,D_k\,\widehat m_k,
\qquad
D_k=\mathrm{diag}\Big(\big(\sqrt{\widehat v_k}+\varepsilon\big)^{-1}\Big).
$$

- 坐标尺度不变性：把某个参数重缩放 $\theta_i\to c\theta_i$，梯度缩放 $1/c$，$m/\sqrt v$ 不变，更新对 $c$ 不敏感——这对嵌入、LayerNorm、注意力矩阵共存的模型很重要；
- 当坐标梯度长期同号时 $\widehat m/\sqrt{\widehat v}\approx\pm1$，表现为每步固定 $\eta$ 的“近似 sign 更新”；
- 与二阶方法的联系：$D_k$ 是对角预条件子；$\varepsilon$ 同时起数值稳定与“信任域半径”作用（$\varepsilon$ 越大越保守）。

### 5.4 AdamW：解耦权重衰减

$$
\theta_k=\theta_{k-1}-\eta_k\Big(\frac{\widehat m_k}{\sqrt{\widehat v_k}+\varepsilon}+\lambda_{\text{wd}}\,\theta_{k-1}\Big).
$$

与“把 $\frac{\lambda}{2}\|\theta\|^2$ 加进损失再交给 Adam”不同，AdamW 的衰减不经过 $m/\sqrt v$ 的缩放，因此大梯度坐标不会被“豁免”衰减。实践中通常不对 LayerNorm 参数、偏置与嵌入施加衰减。

### 5.5 已知的收敛病与修补

- 经典 Adam 在部分凸问题上不收敛（构造性反例），修补是 AMSGrad 等形式；深度学习实践中更常见的是 $\varepsilon$ 与调度的调节；
- 训练早期 $v_k$ 样本少、估计噪声大，更新范数可能很大，这是 warmup（第 7 节）的动机之一；
- 自适应方法并不免费：每个参数要存 $m,v$ 两份状态，3B 模型 fp32 下约 $24$ GB（《10-grad-backprop-optimizer-gpu.md》§6.5），这直接决定必须做优化器状态分片（ZeRO/FSDP）。

---

## 6. 二阶方法与自然梯度

### 6.1 牛顿法

$$
\theta_{k}=\theta_{k-1}-\big(\nabla^2 f(\theta_{k-1})\big)^{-1}\nabla f(\theta_{k-1}).
$$

在局部二次近似下一步到最优；问题：Hessian 是 $d\times d$（3B 参数下不可行）、需要正定化、计算 $O(d^3)$。

### 6.2 自然梯度

把参数空间的距离换成分布空间的距离（KL 的局部二次型是 Fisher 信息 $F(\theta)$）：

$$
\theta_{k}=\theta_{k-1}-\eta\,F(\theta_{k-1})^{-1}\nabla f(\theta_{k-1}).
$$

概率模型的自然梯度不依赖参数化（重参数化不变），与 Adam 的“对角 Fisher 近似”是同一系谱。

### 6.3 实用近似

- Gauss–Newton：用 $J^{\top}J$ 近似 Hessian（最小二乘）；
- K-FAC：按层对 Fisher 做 Kronecker 分解；
- Shampoo：按层用 $\big(\sum gg^{\top}\big)^{1/4}$ 类矩阵预条件。

这些方法在 10 亿参数以上仍不常用：状态与计算代价过高，而 Adam 的性价比在工程上占优。官方 AlphaProof 选择 Adam 与此一致。

---

## 7. 学习率调度与梯度裁剪

### 7.1 常见调度

- **阶跃衰减**：$\eta_k=\eta_0\gamma^{\lfloor k/s\rfloor}$；
- **逆平方根**：$\eta_k=\eta_0/\sqrt{k}$（SGD 理论友好）；
- **warmup + 余弦**：

$$
\eta_k=
\begin{cases}
\eta_{\max}\dfrac{k}{k_{\text{warm}}}, & k\le k_{\text{warm}},\\[6pt]
\eta_{\min}+\dfrac{\eta_{\max}-\eta_{\min}}{2}\Big(1+\cos\Big(\pi\dfrac{k-k_{\text{warm}}}{K-k_{\text{warm}}}\Big)\Big), & k\gt k_{\text{warm}} .
\end{cases}
$$

**为什么 warmup**：训练初期 $m,v$ 的样本少，$\sqrt{\widehat v}$ 偏小，更新 $\eta\,\widehat m/\sqrt{\widehat v}$ 会异常大；线性升温让统计量先稳定下来。

### 7.2 梯度裁剪（全局范数）

约束每步的有效步长，等价于把梯度投影到半径 $c$ 的球内：

$$
g\leftarrow g\cdot\min\Big(1,\ \frac{c}{\|g\|_2}\Big),
\qquad
\|g\|_2=\sqrt{\sum_p\|g_p\|_F^2}.
$$

$c$ 通常取 $1.0$（大模型）或 $0.5$。裁剪后再进 Adam。**官方伪代码没有裁剪**；对 $10^{6}$ 步的 RL 训练，分布漂移会产生偶发大梯度，复现时建议加上并在小规模上验证其收益（《10-grad-backprop-optimizer-gpu.md》§5.4）。

### 7.3 与非凸训练的关系

- 裁剪在几何上定义了一个信任域：$\|\Delta\theta\|\le\eta c$；
- PPO 的 ratio 裁剪、TRPO/自然梯度的 KL 约束，都是同一思想的策略优化版本（与 AlphaProof 无关，但它选择“只用成功轨迹的 CE”来隐式限制策略移动，《6-alpha-proof-mcts-rl.md》§6.2）。

---

## 8. 非凸优化的现实图景

- 深度网络的损失面有大量鞍点与平台；一阶方法能逃离鞍点，随机噪声有正则化与逃逸作用；
- 大批量训练有“临界批量”现象：超过它之后，加大 batch 的收益递减；常用经验规则是 $\eta\propto B$（小批量）或 $\eta\propto\sqrt{B}$（大批量）；
- 泛化与尖锐度/平坦度相关的讨论尚无定论；对本系列而言，需要记住的是 **AlphaProof 的数据分布与损失面都随策略演化而移动**（非平稳），因此调度、裁剪、SFT 混合这些“稳定器”比理论收敛率更重要。

---

## 9. AlphaProof 的优化器事实与方法学小结

- 官方伪代码：`self.optimizer = optax.adam(config.lr)`；更新循环是 `jax.grad` 求梯度、`optimizer.update` 产生更新、`optax.apply_updates` 写回参数（《10-grad-backprop-optimizer-gpu.md》§5.1）；
- 损失权重：策略 $1.0$、价值 $10^{-3}$（Supplementary Table 6）；
- 学习率、$\beta_1,\beta_2,\varepsilon$、调度、裁剪未见公布；公开代码里 `lr=1.0` 是占位；
- 复现建议：Adam 默认超参起步，warmup $5\times10^{3}$ 步加余弦衰减到峰值学习率的 $1/10$，全局范数裁剪 $1.0$，并在相同总步数与 batch 下做小规模扫描（重建）；
- 批大小 4096、约 $10^{6}$ 步对应约 $8\times10^{4}$ TPU-days（其中搜索占主导，learner 占比小）。
