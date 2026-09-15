# MCTS + 强化学习：从贝尔曼方程到 AlphaZero

下面把 PUCT 放在一个有限 MDP 框架中解释。记 MDP 为 $(\mathcal{S},\mathcal{A},P,r,\gamma)$，策略为 $\pi(a\mid s)$，动作价值为

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s,\ a_0=a\right].
$$

AlphaZero/MCTS 中常用的 PUCT 选择公式是

$$
a^\star=\arg\max_{a\in\mathcal{A}(s)}
\left[
Q(s,a)+c_{\text{puct}}\,P(s,a)\,
\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\right].
$$

若记父节点总访问 $N(s)=\sum_b N(s,b)$，则

$$
a^\star=\arg\max_{a\in\mathcal{A}(s)}
\left[
Q(s,a)+c_{\text{puct}}\,P(s,a)\,
\frac{\sqrt{N(s)}}{1+N(s,a)}
\right].
$$

其中：

- $N(s,a)$：在状态 $s$ 选择动作 $a$ 的次数；
- $Q(s,a)$：搜索中得到的动作价值估计，通常是回传回报的平均；
- $P(s,a)$：先验概率，通常来自策略网络 $\pi_\theta(a\mid s)$；
- $c_{\text{puct}}$：探索常数；
- $\sqrt{N(s)}/(1+N(s,a))$：不确定性/探索奖励。

回传时典型更新为

$$
N(s,a)\leftarrow N(s,a)+1,\qquad
W(s,a)\leftarrow W(s,a)+G,\qquad
Q(s,a)=\frac{W(s,a)}{N(s,a)}.
$$

根节点搜索结束后，MCTS 输出访问次数分布

$$
\pi_{\text{MCTS}}(a\mid s_0)
=\frac{N(s_0,a)^{1/\tau}}{\sum_b N(s_0,b)^{1/\tau}},
$$

其中 $\tau$ 是温度。

本文的目标是从数学上完整推导：在 MCTS + 强化学习框架下，下面四件事如何随着策略网络和价值网络参数的更新，逐步提高系统的性能。

1. 贝尔曼期望方程：策略评估为什么会收敛，收敛多快；
2. 贝尔曼最优方程：策略改进为什么不会变差，并且趋向最优策略；
3. 策略梯度下降：一次参数更新究竟带来多少性能提升；
4. AlphaZero 架构：策略网络、价值网络与搜索如何构成一个正反馈闭环。

下面从最基础开始展开。

---

## 1. PUCT 的直观含义

PUCT 是 “Predictor + UCT” 的意思。它把动作选择分成两部分：

$$
\underbrace{Q(s,a)}_{\text{利用项}}
+
\underbrace{c_{\text{puct}}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}}_{\text{探索项}}.
$$

### 1.1 $Q(s,a)$：利用项

$Q(s,a)$ 是当前搜索对动作 $a$ 的价值估计。若某动作过去模拟回报高，则 $Q(s,a)$ 大，PUCT 倾向于继续选择它。

这对应强化学习中的策略改进：选择当前估计价值高的动作。

### 1.2 $P(s,a)$：先验引导

$P(s,a)$ 来自策略网络。它表示“在还没充分搜索前，神经网络认为动作 $a$ 有多好”。

如果没有先验，MCTS 在巨大动作空间中会像均匀探索的 bandit，效率很低。先验让搜索优先尝试看起来有希望的动作。

从策略梯度角度看，$P(s,a)\approx \pi_\theta(a\mid s)$，所以 PUCT 是在当前策略附近做搜索，而不是完全盲目搜索。

### 1.3 $\sqrt{N(s)}/(1+N(s,a))$：探索项

- 当 $N(s,a)=0$ 时，探索项为 $c_{\text{puct}}P(s,a)\sqrt{N(s)}$，先验越高的动作越值得先试。
- 当 $N(s,a)$ 增大时，分母 $1+N(s,a)$ 变大，探索奖励减小。
- 当父节点 $N(s)$ 增大时，说明这个状态被反复考虑，应该继续探索尚未充分访问的动作。

这类似 UCB：

$$
Q(s,a)+c\sqrt{\frac{\log N(s)}{N(s,a)}}.
$$

PUCT 用先验 $P(s,a)$ 替代均匀探索，并用 $\sqrt{N(s)}/(1+N(s,a))$ 作为不确定性代理。

---

## 2. 贝尔曼期望方程：策略评估的数学基础

本节回答第一个问题：给定一个固定策略，价值估计为什么会收敛，以及随着访问次数和迭代次数增加，误差以什么速度变小。

### 2.1 从回报递推推导贝尔曼期望方程

定义折扣回报

$$
G_t=\sum_{k=0}^\infty \gamma^k r_{t+k}.
$$

回报满足一步递推

$$
G_t=r_t+\gamma G_{t+1}.
$$

把它代入动作价值的定义，并用全期望公式：

$$
Q^\pi(s,a)
=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
=\mathbb{E}_\pi[r_t+\gamma G_{t+1}\mid s_t=s,a_t=a].
$$

第一项是即时期望回报 $r(s,a)$；第二项对 $s_{t+1}$ 和 $a_{t+1}$ 展开：

$$
\mathbb{E}_\pi[G_{t+1}\mid s_t=s,a_t=a]
=\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a').
$$

于是得到贝尔曼期望方程

$$
Q^\pi(s,a)
=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a').
$$

若记状态价值

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),
$$

则方程也可以写成更紧凑的两式：

$$
Q^\pi(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s'),
\qquad
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

这套方程的意义是：**价值函数不是独立未知量，而是自身的一步递推的不动点。**

### 2.2 矩阵形式与解析解

把 $Q^\pi$ 看成向量，记 $P^\pi$ 为联合转移矩阵

$$
P^\pi(s,a;s',a')=P(s'\mid s,a)\,\pi(a'\mid s'),
$$

则贝尔曼期望方程是线性方程组

$$
Q^\pi=r+\gamma P^\pi Q^\pi.
$$

在 $\gamma<1$ 时 $I-\gamma P^\pi$ 可逆，因此有解析解

$$
Q^\pi=(I-\gamma P^\pi)^{-1}r=\sum_{k=0}^\infty \gamma^k (P^\pi)^k r.
$$

末式的含义是：价值等于“走 $k$ 步的期望回报”按 $\gamma^k$ 加权求和，这解释了为什么 $\gamma$ 越小估计越容易收敛。

### 2.3 贝尔曼期望算子是压缩映射

定义贝尔曼期望算子 $\mathcal{T}^\pi$：

$$
(\mathcal{T}^\pi Q)(s,a)
=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q(s',a').
$$

**定理.** $\mathcal{T}^\pi$ 在无穷范数下是 $\gamma$-压缩：

$$
\|\mathcal{T}^\pi Q_1-\mathcal{T}^\pi Q_2\|_\infty
\le \gamma\|Q_1-Q_2\|_\infty.
$$

**证明.** 对任意 $(s,a)$，

$$
(\mathcal{T}^\pi Q_1-\mathcal{T}^\pi Q_2)(s,a)
=\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')(Q_1-Q_2)(s',a').
$$

取绝对值并用三角不等式：

$$
|(\mathcal{T}^\pi Q_1-\mathcal{T}^\pi Q_2)(s,a)|
\le \gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')\|Q_1-Q_2\|_\infty
=\gamma\|Q_1-Q_2\|_\infty.
$$

对 $(s,a)$ 取上确界即得结论。∎

由 Banach 不动点定理：$\mathcal{T}^\pi$ 有唯一不动点 $Q^\pi$；并且对任意初值 $Q_0$，迭代 $Q_{k+1}=\mathcal{T}^\pi Q_k$ 满足

$$
\|Q_k-Q^\pi\|_\infty\le \gamma^k\|Q_0-Q^\pi\|_\infty.
$$

取 $Q_0\equiv 0$，并注意 $\|Q^\pi\|_\infty\le R_{\max}/(1-\gamma)$，得到

$$
\|Q_k-Q^\pi\|_\infty\le \gamma^k\frac{R_{\max}}{1-\gamma}.
$$

这就是**策略评估的收敛速度**：每迭代一次，误差至少缩小 $\gamma$ 倍。$\gamma$ 越小，评估收敛越快；$\gamma$ 越接近 $1$，越需要更多迭代。

### 2.4 MCTS 的 Q 是蒙特卡洛策略评估

MCTS 中每个节点 $s$ 的动作 $a$ 维护一个价值估计

$$
Q(s,a)\approx Q^{\pi_{\text{tree}}}(s,a),
$$

其中 $\pi_{\text{tree}}(a\mid s)\propto N(s,a)$ 是树内访问次数诱导的策略。

回传更新

$$
Q_{k+1}(s,a)
=Q_k(s,a)+\frac{1}{N_k(s,a)+1}
\left(G_k-Q_k(s,a)\right)
$$

是 Robbins–Monro 型随机平均：步长取 $\alpha_k=1/(N_k+1)$，满足

$$
\sum_k\alpha_k=\infty,\qquad \sum_k\alpha_k^2<\infty.
$$

若样本 $G_k$ 满足 $\mathbb{E}[G_k\mid s,a]=Q^{\pi_{\text{tree}}}(s,a)$（无偏）且有界方差，则由 Robbins–Monro 定理，

$$
Q_k(s,a)\to Q^{\pi_{\text{tree}}}(s,a)\quad (k\to\infty).
$$

也就是说：**随着访问次数增加，MCTS 先在对当前树策略做策略评估，而且估计越来越准。** 这正是 2.3 节压缩性质的随机版本。

---

## 3. 贝尔曼最优方程与策略改进

本节回答第二个问题：一旦价值估计变准，策略如何改进；为什么改进不会让性能变差；以及为什么反复改进会趋向最优。

### 3.1 最优价值函数

定义最优动作价值和最优状态价值：

$$
Q^*(s,a)=\sup_\pi Q^\pi(s,a),\qquad V^*(s)=\sup_\pi V^\pi(s).
$$

有限 MDP 中上确界可由某确定性平稳策略达到，因此下面用 $\max$ 处理。

### 3.2 贝尔曼最优方程

定义贝尔曼最优算子 $\mathcal{T}$：

$$
(\mathcal{T}Q)(s,a)
=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q(s',a').
$$

**定理.** $\mathcal{T}$ 是 $\gamma$-压缩，其唯一不动点恰好是 $Q^*$，并且

$$
Q^*(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a'),
\qquad
V^*(s)=\max_a Q^*(s,a).
$$

**证明.**

第一步，压缩性。对任意两个向量 $F,G$，有

$$
\left|\max_a F(a)-\max_a G(a)\right|\le \max_a|F(a)-G(a)|.
$$

因此

$$
|(\mathcal{T}F-\mathcal{T}G)(s,a)|
\le \gamma\sum_{s'}P(s'\mid s,a)\max_{a'}|F(s',a')-G(s',a')|
\le \gamma\|F-G\|_\infty,
$$

取上确界得 $\|\mathcal{T}F-\mathcal{T}G\|_\infty\le \gamma\|F-G\|_\infty$。由 Banach 定理，$\mathcal{T}$ 有唯一不动点，记为 $\bar Q$。

第二步，$\bar Q\le Q^*$。定义 $\bar\pi(s)=\arg\max_a\bar Q(s,a)$，则

$$
\mathcal{T}^{\bar\pi}\bar Q=\mathcal{T}\bar Q=\bar Q,
$$

即 $\bar Q$ 是 $\mathcal{T}^{\bar\pi}$ 的不动点，由不动点唯一性 $\bar Q=Q^{\bar\pi}$，所以 $\bar Q\le Q^*$。

第三步，$Q^*\le\bar Q$。对任意策略 $\pi$ 和任意 $Q$，由 $\max_a\ge\sum_a\pi(a\mid s)(\cdot)$ 得逐点不等式

$$
\mathcal{T}^\pi Q\le \mathcal{T}Q.
$$

取 $Q=Q^\pi$：因为 $\mathcal{T}^\pi Q^\pi=Q^\pi$，所以

$$
Q^\pi=\mathcal{T}^\pi Q^\pi\le \mathcal{T}Q^\pi.
$$

由 $\mathcal{T}$ 的单调性可得 $\mathcal{T}^k Q^\pi$ 单调递增，且由压缩性收敛到唯一不动点 $\bar Q$，因此 $Q^\pi\le\bar Q$。对所有 $\pi$ 取上确界得 $Q^*\le\bar Q$。

合并两步得 $Q^*=\bar Q$。∎

### 3.3 值迭代与定量收敛

由压缩性直接得到值迭代

$$
Q_{k+1}=\mathcal{T}Q_k
$$

的误差界

$$
\|Q_k-Q^*\|_\infty\le \gamma^k\|Q_0-Q^*\|_\infty.
$$

更进一步，如果 $\|Q-Q^*\|_\infty\le\varepsilon$，令 $\pi$ 为关于 $Q$ 的贪婪策略，则

$$
\|V^\pi-V^*\|_\infty\le\frac{2\varepsilon}{1-\gamma}.
$$

**证明概要.** 设 $\delta=\max_s(V^*(s)-V^\pi(s))$ 在 $s$ 处取到，则

$$
\delta=V^*(s)-V^\pi(s)
=\max_a Q^*(s,a)-\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

插入并减去 $\sum_a\pi(a\mid s)Q^*(s,a)$，得

$$
\delta\le \left[\max_a Q^*(s,a)-\sum_a\pi(a\mid s)Q^*(s,a)\right]
+\gamma\delta.
$$

方括号一项：由 $\pi$ 对 $Q$ 贪婪以及 $\|Q-Q^*\|_\infty\le\varepsilon$，它不超过 $2\varepsilon$。于是 $\delta\le 2\varepsilon+\gamma\delta$，即 $\delta\le 2\varepsilon/(1-\gamma)$。∎

这说明：**价值估计每准一分，贪婪策略的性能就离最优近一分**，且误差被 $1/(1-\gamma)$ 放大。

### 3.4 策略改进定理（完整证明）

**定理（策略改进）.** 设 $\pi$ 任意，令

$$
\pi'(s)=\arg\max_a Q^\pi(s,a).
$$

则

$$
V^{\pi'}(s)\ge V^\pi(s),\qquad \forall s.
$$

**证明.** 记 $V_0=V^\pi$，并定义 $V_{k+1}=\mathcal{T}^{\pi'}V_k$。

首先，$V_1\ge V_0$：因为

$$
V_1(s)=r(s,\pi'(s))+\gamma\sum_{s'}P(s'\mid s,\pi'(s))V^\pi(s')
=Q^\pi(s,\pi'(s))\ge V^\pi(s).
$$

其次，由 $\mathcal{T}^{\pi'}$ 的单调性，$V_{k+1}\ge V_k$ 蕴含 $V_{k+2}=\mathcal{T}^{\pi'}V_{k+1}\ge\mathcal{T}^{\pi'}V_k=V_{k+1}$。因此 $\{V_k\}$ 单调递增。

最后由压缩性，$V_k\to V^{\pi'}$。于是

$$
V^{\pi'}=\lim_k V_k\ge V_0=V^\pi. \quad\blacksquare
$$

推论：在有限 MDP 中，若每次改进后的策略都严格更优，则由于确定性平稳策略只有有限多个，改进过程必在有限步内到达一个策略，它对自己的 $Q$ 贪婪，从而是最优策略。

### 3.5 PUCT 选择是策略改进

贝尔曼方程告诉我们，若已知 $Q^\pi$，则贪婪策略 $\pi'(s)=\arg\max_a Q^\pi(s,a)$ 不劣于 $\pi$。

PUCT 选择

$$
a^\star=\arg\max_a\left[Q(s,a)+U(s,a)\right]
$$

就是带探索的贪婪策略改进：$Q(s,a)$ 是当前评估，$U(s,a)$ 是乐观探索项。若探索项逐渐减小，则

$$
a^\star\to \arg\max_a Q(s,a).
$$

因此 MCTS 在树内不断执行：

1. 策略评估：用模拟回报更新 $Q(s,a)$；
2. 策略改进：用 PUCT 选择更优动作；
3. 根节点访问分布 $\pi_{\text{MCTS}}$ 是改进后的策略。

这就是近似策略迭代。**每多一次模拟，评估更准；每执行一次 PUCT 选择，策略不劣于当前评估下的贪婪策略。**

### 3.6 乐观贝尔曼算子

PUCT 中的探索项 $U(s,a)$ 可以看作“乐观奖励”。定义乐观贝尔曼算子

$$
(\mathcal{T}^+ Q)(s,a)
=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)
\max_{a'}\left[Q(s',a')+U(s',a')\right].
$$

PUCT 选择正是这个乐观算子的贪婪部分。若 $U(s,a)\ge 0$，则 $\mathcal{T}^+Q\ge\mathcal{T}Q$，即对价值做乐观估计；若 $U(s,a)$ 是上置信界，则 $Q(s,a)+U(s,a)\ge Q^*(s,a)$ 在概率意义下成立。若访问次数趋于无穷且 $U(s,a)\to 0$，则

$$
\mathcal{T}^+\to\mathcal{T},
$$

乐观贝尔曼算子收敛到最优贝尔曼算子，因此 $Q$ 估计趋向 $Q^*$，策略趋向最优策略。

### 3.7 为什么探索项能保证收敛

若所有先验 $P(s,a)>0$，且 $c_{\text{puct}}$ 足够大，则即使某动作当前 $Q(s,a)$ 较低，随着父节点 $N(s)$ 增大，未访问动作的探索项

$$
c_{\text{puct}}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}
$$

最终会超过有界 $Q$ 差，使该动作再次被访问。因此每个动作都会被无限次访问，$Q$ 估计一致。这是 UCB/PUCT 类算法的核心：**无限探索保证价值估计收敛，从而让 3.3、3.4 节的改进定理在极限意义下生效。**

---

## 4. 策略梯度下降：一次参数更新带来多少提升

本节回答第三个问题：当策略由参数 $\theta$ 表示时，更新参数如何直接提高目标函数，以及提高量的显式表达。

### 4.1 目标函数

设策略 $\pi_\theta(a\mid s)$ 可微，初始状态分布为 $\rho_0$，定义

$$
J(\theta)=\mathbb{E}_{s_0\sim\rho_0}\left[V^{\pi_\theta}(s_0)\right].
$$

等价地，若定义折扣访问分布

$$
d^{\pi_\theta}(s)=(1-\gamma)\sum_{t=0}^\infty\gamma^t P(s_t=s),
$$

则

$$
J(\theta)=\frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}
\left[r(s,a)\right].
$$

### 4.2 得分函数技巧

对任意依赖 $\theta$ 的分布 $p_\theta$，有恒等式

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

对轨迹 $\tau=(s_0,a_0,r_0,s_1,\dots)$，其对数概率分解为

$$
\log p_\theta(\tau)
=\log\rho_0(s_0)+\sum_t\log\pi_\theta(a_t\mid s_t)
+\sum_t\log P(s_{t+1}\mid s_t,a_t).
$$

注意环境动力学 $P$ 与 $\rho_0$ 不含 $\theta$，所以

$$
\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

因此

$$
\nabla_\theta J(\theta)
=\nabla_\theta\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]
=\mathbb{E}_{\tau\sim p_\theta}\left[R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right],
$$

其中 $R(\tau)=\sum_t\gamma^t r_t$。

### 4.3 策略梯度定理（完整推导）

**定理.** 在适当正则条件下，

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\,Q^{\pi_\theta}(s,a)\right].
$$

**证明.** 从 4.2 的结果出发：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_{t}\gamma^t r_t\sum_{t'}\nabla_\theta\log\pi_\theta(a_{t'}\mid s_{t'})\right].
$$

利用因果性：当 $t'>t$ 时，$\nabla_\theta\log\pi_\theta(a_{t'}\mid s_{t'})$ 在给定 $s_{t'}$ 后对 $a_{t'}\sim\pi_\theta$ 的条件期望为

$$
\sum_a\pi_\theta(a\mid s_{t'})\nabla_\theta\log\pi_\theta(a\mid s_{t'})
=\nabla_\theta\sum_a\pi_\theta(a\mid s_{t'})=0,
$$

而 $r_t$（$t<t'$）不依赖未来的 $a_{t'}$，故这些交叉项期望为零。保留 $t'\le t$ 的项：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\sum_{k\ge t}\gamma^k r_k\right].
$$

令 $G_t=\sum_{k\ge t}\gamma^{k-t}r_k$，则

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_{t}\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right].
$$

再由 $\mathbb{E}[G_t\mid s_t,a_t]=Q^{\pi_\theta}(s_t,a_t)$，并整理折扣访问分布，即得

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\,Q^{\pi_\theta}(s,a)\right].
\quad\blacksquare
$$

直观上，策略梯度让高价值动作的概率增加，低价值动作的概率减少。

### 4.4 基线不变性与优势函数

对任意只依赖状态、不依赖动作的基线 $b(s)$：

$$
\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\right]
=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)
=b(s)\nabla_\theta 1=0.
$$

所以基线不改变梯度的期望，但能显著降低方差。取 $b(s)=V^{\pi_\theta}(s)$，得到优势形式

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a\mid s)\,A^{\pi_\theta}(s,a)\right],
\qquad
A^{\pi_\theta}(s,a)=Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s).
$$

### 4.5 一次更新为什么提高性能：性能差引理

**引理（性能差引理）.** 对任意两个策略 $\pi,\pi'$ 和任意初始状态 $s_0$，

$$
V^{\pi'}(s_0)-V^{\pi}(s_0)
=\mathbb{E}_{\tau\sim p_{\pi'}}
\left[\sum_{t=0}^\infty\gamma^t A^{\pi}(s_t,a_t)\right],
$$

等价的访问分布形式为

$$
J(\pi')-J(\pi)
=\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d^{\pi'},\,a\sim\pi'}\left[A^{\pi}(s,a)\right].
$$

**证明.** 沿一条轨迹逐项展开优势函数：

$$
\sum_{t=0}^\infty\gamma^t A^\pi(s_t,a_t)
=\sum_{t=0}^\infty\gamma^t\left(r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)\right).
$$

右侧求和望远镜式消去：

$$
\sum_{t=0}^\infty\gamma^t r_t+\sum_{t=0}^\infty\left(\gamma^{t+1}V^\pi(s_{t+1})-\gamma^t V^\pi(s_t)\right)
=G_0-V^\pi(s_0).
$$

在轨迹分布 $p_{\pi'}$ 下取期望，左边即为引理内容。∎

推论（改进下界）：如果对所有 $s$ 都有

$$
\sum_a\pi'(a\mid s)A^{\pi}(s,a)\ge 0,
$$

则 $V^{\pi'}\ge V^{\pi}$。这说明：**只要新策略在每一步都提升相对旧策略的优势，性能就不降。**

### 4.6 梯度上升：一步提升多少

设 $J$ 可微且 $\nabla J$ 是 $L$-Lipschitz 的，则对任意 $\theta'$，

$$
J(\theta')\ge J(\theta)+\langle\nabla J(\theta),\theta'-\theta\rangle
-\frac{L}{2}\|\theta'-\theta\|^2.
$$

取梯度上升一步

$$
\theta'=\theta+\alpha\nabla J(\theta),\qquad \alpha=\frac{1}{L},
$$

得到显式的改进量

$$
J(\theta')\ge J(\theta)+\frac{1}{2L}\|\nabla J(\theta)\|^2.
$$

也就是说，只要梯度非零，一步更新必有正提升，提升量与梯度范数平方成正比。对随机梯度上升，用步长 $\alpha_t\propto 1/\sqrt{t}$ 可得

$$
\frac{1}{T}\sum_{t=1}^T\mathbb{E}\|\nabla J(\theta_t)\|^2
=\mathcal{O}\left(\frac{1}{\sqrt{T}}\right),
$$

即参数更新次数越多，越接近平稳点。

### 4.7 KL 正则策略改进：镜像下降的完整求解

考虑对每个状态 $s$ 求解

$$
\pi_{k+1}(\cdot\mid s)
=\arg\max_{\pi(\cdot\mid s)}
\left[
\sum_a\pi(a\mid s)Q^{\pi_k}(s,a)
-\frac{1}{\eta}\mathrm{KL}\left(\pi(\cdot\mid s)\,\|\,\pi_k(\cdot\mid s)\right)
\right],
$$

约束 $\sum_a\pi(a\mid s)=1$。构造拉格朗日函数，对 $\pi(a\mid s)$ 求导并令其为零：

$$
Q^{\pi_k}(s,a)-\frac{1}{\eta}\left(\log\frac{\pi(a\mid s)}{\pi_k(a\mid s)}+1\right)-\lambda=0.
$$

于是

$$
\pi(a\mid s)\propto\pi_k(a\mid s)\exp\left(\eta Q^{\pi_k}(s,a)\right).
$$

这就是镜像下降/自然策略梯度式的策略改进：**高 $Q$ 的动作概率指数级增大，但被 KL 惩罚限制在旧策略附近，步长由 $\eta$ 控制。**

### 4.8 AlphaZero 训练策略网络：交叉熵梯度等价于优势加权策略梯度

AlphaZero 中，MCTS 在状态 $s$ 产生访问分布 $\pi_{\text{MCTS}}(a\mid s)$。策略网络 $\pi_\theta$ 通过交叉熵拟合它：

$$
\mathcal{L}_\pi(\theta)
=-\mathbb{E}_s\left[\sum_a\pi_{\text{MCTS}}(a\mid s)\log\pi_\theta(a\mid s)\right].
$$

其梯度为

$$
\nabla_\theta\mathcal{L}_\pi(\theta)
=-\mathbb{E}_s\left[\sum_a\pi_{\text{MCTS}}(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

若搜索给出的目标是 4.7 节形式的改进策略

$$
\pi_{\text{MCTS}}(a\mid s)\approx\frac{\pi_\theta(a\mid s)\exp(\beta A^{\pi_\theta}(s,a))}{Z(s)},
$$

代入并把期望改写为对 $\pi_\theta$ 采样（重要性权重 $w(a)=e^{\beta A}/Z$）：

$$
-\nabla_\theta\mathcal{L}_\pi(\theta)
=\mathbb{E}_{a\sim\pi_\theta}\left[w(a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

对指数做一阶展开 $e^{\beta A}\approx 1+\beta A$，利用基线不变性去掉常数项 1，即得

$$
-\nabla_\theta\mathcal{L}_\pi(\theta)
\approx\frac{\beta}{Z}\,
\mathbb{E}_{a\sim\pi_\theta}\left[A^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right]
\propto \nabla_\theta J(\theta).
$$

也就是说，**用 MCTS 分布做交叉熵蒸馏，在小步长意义下与策略梯度上升方向一致**：高优势动作被加权放大，低优势动作被压制。

### 4.9 PUCT 各部分的策略梯度解释

- $Q(s,a)$：价值/优势估计。加常数 $V(s)$ 不影响 $\arg\max$，所以 PUCT 等价于最大化 $A(s,a)+U(s,a)$，偏向正优势动作。
- $P(s,a)$：当前策略先验，相当于策略梯度中的信任域，防止搜索偏离当前策略太远。
- 探索项：控制方差。访问少的动作 $Q$ 估计方差大，PUCT 给它更大探索奖励，减少策略梯度估计的偏差。
- 访问分布 $\pi_{\text{MCTS}}$：策略改进目标，训练策略网络拟合它，相当于优势加权最大似然。

---

## 5. AlphaZero 架构：网络、损失与闭环提升

本节回答第四个问题：策略网络和价值网络如何组织，损失如何分解，以及为什么每轮参数更新会把整体性能推高。

### 5.1 网络结构

AlphaZero 用一套共享表示加两个输出头：

- 共享躯干 $h_\theta(s)\in\mathbb{R}^d$（棋盘类任务里是残差卷积塔，输入是状态 $s$ 的特征编码）；
- 策略头：输出 logits $f_\theta(s)$，经 softmax 得到 $p_\theta(a\mid s)=\frac{\exp f_\theta(s)_a}{\sum_b\exp f_\theta(s)_b}$；
- 价值头：输出标量 $v_\theta(s)=\tanh(g_\theta(s))\in(-1,1)$，用于估计当前策略下状态 $s$ 的胜负期望。

直觉分工：策略头给出“先搜哪里”的先验 $P$，价值头给出“叶子节点值多少”的估计，两者都直接进入 PUCT 搜索。

### 5.2 搜索给出训练目标

从状态 $s$ 出发做一次 MCTS，得到：

- 访问分布 $\pi_{\text{MCTS}}(a\mid s)$，作为策略目标的“软标签”；
- 自对弈结束后得到胜负结果 $z\in\{-1,0,+1\}$，作为价值目标的蒙特卡洛样本。

对当前策略 $\pi_{\text{MCTS}}$ 而言，$z$ 是无偏的价值样本：

$$
\mathbb{E}[z\mid s]=V^{\pi_{\text{MCTS}}}(s).
$$

### 5.3 损失函数与梯度分解

AlphaZero 的训练损失（忽略正则项细节）为

$$
L(\theta)
=\underbrace{\mathbb{E}_{(s,z)}\left[\left(z-v_\theta(s)\right)^2\right]}_{\text{价值回归}}
+\underbrace{\mathbb{E}_s\left[-\sum_a\pi_{\text{MCTS}}(a\mid s)\log p_\theta(a\mid s)\right]}_{\text{策略交叉熵}}
+c\|\theta\|_2^2.
$$

两部分的梯度分别是：

$$
\nabla_\theta L_{\text{value}}
=\mathbb{E}\left[-2\left(z-v_\theta(s)\right)\nabla_\theta v_\theta(s)\right],
$$

$$
\nabla_\theta L_{\text{policy}}
=\mathbb{E}_s\left[-\sum_a\pi_{\text{MCTS}}(a\mid s)\nabla_\theta\log p_\theta(a\mid s)\right].
$$

- 价值项是蒙特卡洛策略评估：最小化均方误差会让 $v_\theta(s)$ 逼近 $\mathbb{E}[z\mid s]=V^{\pi_{\text{MCTS}}}(s)$。也就是说，**价值网络拟合的是“搜索改进后策略”的价值函数**。
- 策略项是 4.8 节证明过的优势加权策略梯度方向：**策略网络被拉向搜索产生的更优分布**。

参数更新同时作用在两个头上，而两个头的输出又同时决定下一轮搜索的质量，这就是闭环的数学来源。

### 5.4 性能为什么随参数更新逐步提高：闭环论证

把训练轮次记作 $k$，整个系统是一个迭代：

$$
\pi_{\theta_k}
\;\longrightarrow\;
\left(P_{\theta_k},v_{\theta_k}\right)
\;\longrightarrow\;
\text{PUCT 搜索}
\;\longrightarrow\;
\left(\pi^{k}_{\text{MCTS}},Q_k\right)
\;\longrightarrow\;
\theta_{k+1}.
$$

**第一步：网络更好，搜索更准。** 先验 $P$ 越接近真实最优策略，PUCT 越少浪费在坏动作上；叶子价值 $v_\theta$ 越准，回传的 $Q$ 偏差越小。极端情形下，如果 $P=\pi^*$、$v_\theta=V^*$，一次搜索就能识别最优动作，$Q$ 的回传误差为 0。

**第二步：搜索更准，改进目标更大。** 由性能差引理，搜索产生的策略相对当前策略的提升是

$$
J(\pi^{k}_{\text{MCTS}})-J(\pi_{\theta_k})
=\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d^{\pi^{k}_{\text{MCTS}}}}
\left[\sum_a\pi^{k}_{\text{MCTS}}(a\mid s)A^{\pi_{\theta_k}}(s,a)\right].
$$

搜索越准（$Q_k$ 越接近 $Q^{\pi_{\theta_k}}$），这个期望优势越大；当 $\pi^{k}_{\text{MCTS}}$ 趋于对 $Q_k$ 贪婪时，提升量趋于最大。

**第三步：网络拟合更好的目标，下一轮起点更高。** 交叉熵项让 $p_{\theta_{k+1}}\to\pi^{k}_{\text{MCTS}}$，价值项让 $v_{\theta_{k+1}}\to V^{\pi^{k}_{\text{MCTS}}}$。于是下一轮搜索的先验和叶子价值都基于一个更好的策略，形成正反馈。

在理想化条件下，这一迭代有严格的单调性保证：

**命题.** 设状态-动作空间有限，假设每轮搜索能精确计算 $Q^{\pi_k}$ 并令下一轮策略为

$$
\pi_{k+1}(s)=\arg\max_a Q^{\pi_k}(s,a),
$$

且网络能精确表示该策略与其价值，则

$$
V^{\pi_{k+1}}(s)\ge V^{\pi_k}(s)\quad \forall s,
$$

并且经过有限轮后策略达到最优。

**证明.** 单调性由策略改进定理（3.4 节）直接给出。有限轮终止：有限 MDP 中确定性平稳策略只有有限多个，且每轮策略都是对自己的 $Q$ 贪婪；一旦策略不再变化，它满足贪婪最优性条件，等价于贝尔曼最优方程的不动点，因此是最优策略（3.2 节）。∎

实际系统中存在函数逼近误差、采样噪声、探索温度等偏差，严格单调性不成立；但 4.5 节的下界和 4.6 节的梯度上升分析说明：**只要每轮更新的方向正确、步长足够小，改进量与误差项可控时，性能趋势向上；训练轮数越多，策略越接近搜索给出的目标，目标本身又随搜索变好而提高。**

### 5.5 与贝尔曼方程的接口：树内引导

MCTS 的回传与贝尔曼方程的关系可以写成更一般的形式。对访问路径上每个节点 $(s,a)$，若叶子节点用价值网络估计，则备份为

$$
Q(s,a)\leftarrow
\frac{1}{N(s,a)}\sum_{m=1}^{N(s,a)}
\left(\sum_{i=0}^{d_m-1}\gamma^i r_i+\gamma^{d_m}v_\theta(s_m)\right),
$$

其中 $d_m$ 是第 $m$ 次模拟从 $(s,a)$ 到叶子的深度。

- 若 $v_\theta=V^{\pi}$ 精确且 $N(s,a)\to\infty$，则 $Q(s,a)\to Q^{\pi}(s,a)$，这正是贝尔曼期望方程的不动点条件：**树内备份在极限意义下解贝尔曼期望方程。**
- 若不用价值网络而用完整蒙特卡洛回报，则 $Q$ 收敛到 $\pi_{\text{MCTS}}$ 的自对弈价值，是“零引导”的另一端。
- AlphaZero 实际处于两者之间：用 $v_\theta$ 截断深度 $d_m$，相当于有限深度 Bellman 备份，方差更小，代价是引入了网络逼近误差。

这也解释了为什么价值网络质量会直接影响搜索强度：**$v_\theta$ 是贝尔曼备份的边界条件，边界错多少，回传的 $Q$ 就错多少。**

---

## 6. 总结

1. **贝尔曼期望方程（策略评估）**

   - 从回报递推 $G_t=r_t+\gamma G_{t+1}$ 出发，可推出 $Q^\pi$ 满足 $Q^\pi=\mathcal{T}^\pi Q^\pi$；
   - $\mathcal{T}^\pi$ 是 $\gamma$-压缩，迭代误差按 $\gamma^k$ 衰减；
   - MCTS 的 $Q$ 回传是 Robbins–Monro 随机平均，访问次数增加时收敛到当前树策略的价值；
   - 因此参数更新（搜索加深）首先让评估更准，这是后续一切改进的前提。

2. **贝尔曼最优方程（策略改进）**

   - 最优算子 $\mathcal{T}$ 是 $\gamma$-压缩，唯一不动点即 $Q^*$；
   - 策略改进定理保证：对 $Q$ 贪婪的策略满足 $V^{\pi'}\ge V^\pi$；
   - 值迭代误差 $\gamma^k$ 衰减，贪婪策略的性能误差不超过 $2\varepsilon/(1-\gamma)$；
   - PUCT 是带探索的贪婪改进（乐观贝尔曼算子），探索项趋于 0 时趋向最优贝尔曼算子；
   - 因此“更准的 $Q$”直接转化为“不劣、且逐步逼近最优的策略”。

3. **策略梯度下降（参数更新的显式提升）**

   - 得分函数技巧 + 因果性给出策略梯度定理
     $\nabla J=\mathbb{E}[\nabla\log\pi\,A^\pi]$；
   - 基线不改变期望，只降方差；
   - 性能差引理给出 $J(\pi')-J(\pi)=\frac{1}{1-\gamma}\mathbb{E}_{d^{\pi'}}[A^\pi]$，说明优势为正即提升；
   - 平滑性下梯度上升一步的提升量至少为 $\frac{1}{2L}\|\nabla J\|^2$；
   - KL 正则改进是 $\pi_{k+1}\propto\pi_k\exp(\eta Q)$，AlphaZero 的交叉熵蒸馏在小步长下与策略梯度同向。

4. **AlphaZero 架构（闭环提升）**

   - 共享躯干 + 策略头 + 价值头；损失 = 价值 MSE + 策略交叉熵；
   - 价值头回归 $z$，目标是 $V^{\pi_{\text{MCTS}}}$；策略头拟合 $\pi_{\text{MCTS}}$，方向是优势加权策略梯度；
   - 闭环：
     $\pi_\theta\to$ PUCT 搜索 $\to(\pi_{\text{MCTS}},Q)\to$ 网络训练 $\to\theta'$，
     下一轮搜索从更好的先验和值函数开始；
   - 理想条件下每轮策略改进不降且有限步到最优；实际中每轮更新带来“搜索更准 + 目标更好”的一阶提升，误差项由网络逼近和采样噪声控制。

5. **一句话概括**

   PUCT 不是孤立的启发式公式，而是“带先验的乐观 bandit + 贝尔曼策略评估 + 策略改进 + 策略梯度式策略拟合”的合成；策略网络和价值网络参数的每一次更新，都在同一条链上同时改善“评估精度”和“改进方向”，使性能逐轮提高。
