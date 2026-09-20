# 11 概率、极大似然与机器学习估计理论

参考：本系列《6-alpha-proof-mcts-rl.md》《9-mainrl-ttrl-llm-value-head.md》《10-grad-backprop-optimizer-gpu.md》与 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/` 中的 `12-rl-math.md`。

定位：这是全套笔记的“概率地基”。AlphaProof 的两个损失（策略交叉熵、价值分类交叉熵）本质上都是极大似然估计；价值头逼近的条件期望 $\mathbb{E}[G\mid s]$ 也是估计论对象。本文把这些概念从零推起；下一批文档（12–15）补齐 Softmax/MLP、优化器、反向传播与数学工具。

---

## 1. 概率语言回顾

### 1.1 随机变量与分布

概率空间 $(\Omega,\mathcal{F},\mathbb{P})$。随机变量 $X:\Omega\to\mathcal{X}$ 把随机结果映射到取值空间。分布函数与（离散）概率质量函数 /（连续）概率密度：

$$
F_X(x)=\mathbb{P}(X\le x),
\qquad
p_X(x)=\mathbb{P}(X=x)\quad\text{或}\quad f_X(x)=\frac{\mathrm{d}F_X}{\mathrm{d}x}.
$$

期望、方差、协方差：

$$
\mathbb{E}[X]=\int x\,\mathrm{d}F_X(x),
\qquad
\mathrm{Var}(X)=\mathbb{E}\big[(X-\mathbb{E}X)^2\big]=\mathbb{E}[X^2]-\big(\mathbb{E}[X]\big)^2,
$$

$$
\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mathbb{E}X)(Y-\mathbb{E}Y)\big].
$$

### 1.2 条件期望与全期望公式

条件期望 $\mathbb{E}[X\mid Y]$ 是 $Y$ 的函数（一个随机变量），满足对任意可测函数 $g$：

$$
\mathbb{E}\big[X\,g(Y)\big]=\mathbb{E}\Big[\mathbb{E}[X\mid Y]\,g(Y)\Big].
$$

**全期望公式（塔性质）：**

$$
\mathbb{E}\big[\mathbb{E}[X\mid Y]\big]=\mathbb{E}[X].
$$

证明（离散情形）：设 $Y$ 取值 $y$，则

$$
\mathbb{E}\Big[\mathbb{E}[X\mid Y]\Big]
=\sum_y \mathbb{P}(Y=y)\,\mathbb{E}[X\mid Y=y]
=\sum_y \mathbb{P}(Y=y)\sum_x x\,\mathbb{P}(X=x\mid Y=y)
=\sum_x x\,\mathbb{P}(X=x).
$$

### 1.3 条件期望是最优的最小二乘预测器

**命题.** 对任意可测 $g$，

$$
\mathbb{E}\big[(X-g(Y))^2\big]
\;\ge\;
\mathbb{E}\big[(X-\mathbb{E}[X\mid Y])^2\big].
$$

**证明.** 对固定的 $Y=y$，令 $m(y)=\mathbb{E}[X\mid Y=y]$。展开：

$$
\mathbb{E}\big[(X-g(Y))^2\mid Y=y\big]
=\mathbb{E}\big[(X-m(y))^2\mid Y=y\big]+\big(m(y)-g(y)\big)^2
+2\big(m(y)-g(y)\big)\underbrace{\mathbb{E}\big[X-m(y)\mid Y=y\big]}_{=0}.
$$

三项中第一项与 $g$ 无关，第三项为零，第二项非负。对 $y$ 按 $Y$ 的分布取期望即得。$\square$

这条命题是价值网络的概率语义：**给定状态 $s$ 预测回报 $G$ 的最优预测器就是条件期望 $\mathbb{E}[G\mid s]$**，也就是价值函数 $V(s)$（《6-alpha-proof-mcts-rl.md》§1.1 的 $V(s)=-d(s)$）。

### 1.4 独立同分布样本

样本 $x_1,\dots,x_n$ 独立同分布（i.i.d.）于 $P$，记经验分布

$$
\widehat P_n=\frac{1}{n}\sum_{i=1}^{n}\delta_{x_i}.
$$

对任意有界 $f$，由大数定律

$$
\frac{1}{n}\sum_{i=1}^{n}f(x_i)\ \to\ \mathbb{E}_P[f(X)]
\qquad\text{几乎必然}.
$$

这条收敛性在后面解释“最小化平均对数似然为什么等于最小化 KL”时会用到。

---

## 2. 参数模型与极大似然估计

### 2.1 似然、对数似然、得分函数

设数据 $x$ 来自参数化分布族 $\{p_\theta:\theta\in\Theta\}$。给定观测 $x$，把 $p_\theta(x)$ 看成 $\theta$ 的函数称为**似然**：

$$
L(\theta)=p_\theta(x),
\qquad
\ell(\theta)=\log L(\theta)=\log p_\theta(x).
$$

i.i.d. 样本 $D=(x_1,\dots,x_n)$ 的联合似然与对数似然：

$$
L(\theta;D)=\prod_{i=1}^{n}p_\theta(x_i),
\qquad
\ell(\theta;D)=\sum_{i=1}^{n}\log p_\theta(x_i).
$$

**得分函数（score）**与**观测信息**：

$$
s(\theta)=\nabla_\theta\,\ell(\theta;D),
\qquad
J(\theta)=-\nabla^2_\theta\,\ell(\theta;D)\quad(\text{观测 Fisher 信息的样本版}).
$$

### 2.2 极大似然估计的定义

$$
\widehat\theta_{\mathrm{MLE}}
=\arg\max_{\theta\in\Theta}\ \ell(\theta;D).
$$

当 $\ell$ 可微且最大值在内部时，一阶条件

$$
\nabla_\theta \ell(\theta;D)\Big|_{\theta=\widehat\theta}=0
$$

是估计的**似然方程**。注意把 $D$ 视为随机时，$\widehat\theta(D)$ 本身是随机变量——这是估计理论讨论偏差与方差的出发点。

---

## 3. 三个完全展开的例子

### 3.1 Bernoulli 模型

$X\in\{0,1\}$，$p_\theta(1)=\theta$。对数似然：

$$
\ell(\theta)=\sum_{i=1}^{n}\Big[x_i\log\theta+(1-x_i)\log(1-\theta)\Big].
$$

求导并令零：

$$
\frac{\mathrm{d}\ell}{\mathrm{d}\theta}
=\frac{\sum_i x_i}{\theta}-\frac{n-\sum_i x_i}{1-\theta}=0
\;\Longrightarrow\;
\widehat\theta=\frac{1}{n}\sum_i x_i=\bar x .
$$

二阶导 $-\big(\sum x_i/\theta^2+(n-\sum x_i)/(1-\theta)^2\big)\lt 0$，故为最大值。

### 3.2 Categorical 模型（Softmax 的 MLE）

$Y\in\{1,\dots,K\}$，$\mathbb{P}(Y=k)=p_k$，约束 $\sum_k p_k=1$。令 $N_k=\#\{i:y_i=k\}$，对数似然加 Lagrange 乘子：

$$
\ell(p,\lambda)=\sum_{k=1}^{K}N_k\log p_k+\lambda\Big(1-\sum_k p_k\Big).
$$

对 $p_k$ 求导：

$$
\frac{\partial \ell}{\partial p_k}=\frac{N_k}{p_k}-\lambda=0
\;\Longrightarrow\;
p_k=\frac{N_k}{\lambda}.
$$

用约束 $\sum_k N_k=n$ 得 $\lambda=n$，即

$$
\widehat p_k=\frac{N_k}{n}\quad\text{（经验频率）}.
$$

**与交叉熵的联系.** 若用参数化 $p=\mathrm{softmax}(z)$ 直接对 $z$ 求导，用链式法则与 softmax 的 Jacobian（见第 12 篇）：

$$
\frac{\partial \ell}{\partial z_k}
=\sum_{j}\frac{N_j}{p_j}\frac{\partial p_j}{\partial z_k}
=\sum_j \frac{N_j}{p_j}\,p_j\big(\delta_{jk}-p_k\big)
=N_k-n\,p_k .
$$

令其为零恰好给出 $p_k=N_k/n$。除以 $n$ 得

$$
\frac{1}{n}\frac{\partial \ell}{\partial z_k}
=\frac{N_k}{n}-p_k
=\underbrace{\text{经验频率}}_{=\ \text{one-hot 目标的平均}}-p_k,
$$

即交叉熵损失的梯度是“预测概率减经验频率”（见第 12 篇 §3、第 10 篇 §1.4 的 $\delta_z=p-e_y$）。**价值头训练就是这个式子：把 64 个 bin 上的预测分布拉向搜索给出回报的经验分布**，其解是条件期望 $\mathbb{E}[G\mid s]=-d(s)$。

### 3.3 正态模型与偏差

$X\sim\mathcal{N}(\mu,\sigma^2)$。对数似然：

$$
\ell(\mu,\sigma^2)
=-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2.
$$

对 $\mu$：$\partial\ell/\partial\mu=\frac{1}{\sigma^2}\sum_i(x_i-\mu)=0$ 给出 $\widehat\mu=\bar x$。对 $\sigma^2$ 令导数为零：

$$
-\frac{n}{2\sigma^2}+\frac{1}{2\sigma^4}\sum_i(x_i-\bar x)^2=0
\;\Longrightarrow\;
\widehat\sigma^2=\frac{1}{n}\sum_i(x_i-\bar x)^2 .
$$

**偏差计算.** 这是有偏估计，因为

$$
\mathbb{E}\Big[\sum_i(x_i-\bar x)^2\Big]=(n-1)\sigma^2
\;\Longrightarrow\;
\mathbb{E}[\widehat\sigma^2]=\frac{n-1}{n}\sigma^2 .
$$

无偏版本 $\frac{1}{n-1}\sum_i(x_i-\bar x)^2$ 说明了估计理论的一个基本事实：**MLE 不必无偏**，渐近意义上它是最优的（第 6 节）。

---

## 4. 从 MLE 到交叉熵与 KL 散度

### 4.1 定义

$$
H(p)=-\sum_x p(x)\log p(x)
\quad(\text{熵}),
$$

$$
H(p,q)=-\sum_x p(x)\log q(x)
\quad(\text{交叉熵}),
$$

$$
D(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
\quad(\text{KL 散度}).
$$

三者满足恒等式

$$
H(p,q)=H(p)+D(p\,\|\,q).
$$

### 4.2 Gibbs 不等式与 KL 非负性

**命题.** $D(p\,\|\,q)\ge 0$，等号当且仅当 $p=q$。

**证明.** 用 Jensen 不等式（对数上凸）：由于 $-\log$ 是凸函数，

$$
D(p\,\|\,q)=\sum_x p(x)\Big(-\log\frac{q(x)}{p(x)}\Big)
\ \ge\ -\log\Big(\sum_x p(x)\frac{q(x)}{p(x)}\Big)
=-\log 1=0 .
$$

$\square$

KL 非对称，不是距离；但它度量了“用 $q$ 编码 $p$”的额外代价（香农编码的语言），这解释了交叉熵损失的几何意义。

### 4.3 平均对数似然收敛到负交叉熵

设真实分布为 $P$，模型为 $p_\theta$。对任意固定的 $\theta$，由大数定律

$$
\frac{1}{n}\ell(\theta;D)
=\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(x_i)
\ \to\ \mathbb{E}_{P}\big[\log p_\theta(X)\big]
=-H(P,p_\theta)
=-H(P)-D(P\,\|\,p_\theta)
$$

几乎必然。由于 $H(P)$ 与 $\theta$ 无关，

$$
\arg\max_\theta\ \mathbb{E}_P[\log p_\theta(X)]
=\arg\min_\theta\ D(P\,\|\,p_\theta).
$$

**结论：MLE 在总体版本上等价于寻找与数据分布 KL 最近的模型；最小化交叉熵 = 最小化 KL = 最大化似然。** 这正是 AlphaProof 用交叉熵训练策略头和价值头的理论根据。

---

## 5. MAP、正则化与贝叶斯视角

### 5.1 最大后验

后验分布

$$
p(\theta\mid D)=\frac{p(D\mid\theta)\,p(\theta)}{p(D)}
\propto p(D\mid\theta)\,p(\theta).
$$

最大后验估计（MAP）：

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta\ \Big[\underbrace{\ell(\theta;D)}_{\text{对数似然}}+\underbrace{\log p(\theta)}_{\text{对数先验}}\Big].
$$

### 5.2 先验即正则化

- 高斯先验 $\theta\sim\mathcal{N}(0,\tau^2 I)$：

$$
\log p(\theta)=-\frac{\|\theta\|^2}{2\tau^2}+\text{const},
$$

于是 MAP 等价于带 $L_2$ 正则（权重衰减）的 MLE，正则系数 $\lambda_{\text{wd}}=1/(2\tau^2)$：

$$
\widehat\theta=\arg\max_\theta\ \Big[\ell(\theta;D)-\frac{1}{2\tau^2}\|\theta\|^2\Big].
$$

- Laplace 先验 $p(\theta_i)=\frac{1}{2b}e^{-|\theta_i|/b}$ 给出 $L_1$ 正则，导致稀疏解。

**注意区分两件事**：AdamW 里的 weight decay 是“更新规则层”的解耦惩罚（第 13 篇 §5），而这里的 $L_2$ 正则是“损失层”的概率先验。官方 AlphaProof 伪代码用的是 `optax.adam`（无 weight decay），论文也没有报告显式正则项，抗遗忘主要靠 10% 的 SFT 混合（《9-mainrl-ttrl-llm-value-head.md》§2.3.6）。

---

## 6. 估计理论的核心定理

### 6.1 偏差、方差与均方误差

对估计量 $\widehat\theta$：

$$
\mathrm{Bias}(\widehat\theta)=\mathbb{E}[\widehat\theta]-\theta,
\qquad
\mathrm{Var}(\widehat\theta)=\mathbb{E}\big[(\widehat\theta-\mathbb{E}\widehat\theta)^2\big].
$$

**分解定理（MSE = 偏差平方 + 方差）：**

$$
\mathbb{E}\big[(\widehat\theta-\theta)^2\big]
=\mathrm{Bias}(\widehat\theta)^2+\mathrm{Var}(\widehat\theta).
$$

**证明.** 记 $\bar\theta=\mathbb{E}[\widehat\theta]$，则

$$
\widehat\theta-\theta=(\widehat\theta-\bar\theta)+(\bar\theta-\theta).
$$

第一项零均值；两交叉项期望为零；平方期望得 $\mathrm{Var}+\mathrm{Bias}^2$。$\square$

### 6.2 一致性与渐近正态

- **相合性**：$\widehat\theta_n\to\theta$（依概率）当 $n\to\infty$；
- **渐近正态性**：在正则条件下，MLE 满足

$$
\sqrt{n}\,\big(\widehat\theta_n-\theta\big)
\ \xrightarrow{\ d\ }\
\mathcal{N}\Big(0,\ I(\theta)^{-1}\Big),
$$

其中 $I(\theta)$ 是 Fisher 信息（下节）。这给出了置信区间与“$n^{-1/2}$ 收敛速率”的来源。

### 6.3 Fisher 信息与 Cramér–Rao 下界

**定义.**

$$
I(\theta)=\mathbb{E}_\theta\Big[\big(\nabla_\theta\log p_\theta(X)\big)\big(\nabla_\theta\log p_\theta(X)\big)^{\top}\Big].
$$

**命题（得分零均值与信息恒等式）.** 在可在积分号下求导的正则条件下：

$$
\mathbb{E}_\theta\big[\nabla_\theta\log p_\theta(X)\big]=0,
\qquad
I(\theta)=-\mathbb{E}_\theta\big[\nabla^2_\theta\log p_\theta(X)\big].
$$

**证明.** 第一式：

$$
\mathbb{E}_\theta[\nabla\log p_\theta]
=\int \frac{\nabla p_\theta}{p_\theta}\,p_\theta\,\mathrm{d}x
=\nabla\int p_\theta\,\mathrm{d}x
=\nabla 1=0 .
$$

第二式：$\nabla^2\log p_\theta=\dfrac{\nabla^2 p_\theta}{p_\theta}-\big(\nabla\log p_\theta\big)\big(\nabla\log p_\theta\big)^{\top}$，取期望，第一项为 $\nabla^2\int p_\theta=0$，第二项是 $-I(\theta)$。$\square$

**Cramér–Rao 下界（叙述与推导要点）.** 对无偏估计量 $\widehat\theta$：

$$
\mathrm{Var}(\widehat\theta)\ \ge\ \frac{1}{n\,I(\theta)}
\quad(\text{单参数情形}).
$$

要点：无偏性给 $\mathbb{E}[\widehat\theta]=\theta$，两边对 $\theta$ 求导得 $\mathrm{Cov}(\widehat\theta,s(\theta))=1$；对 $\mathrm{Cov}$ 用 Cauchy–Schwarz，结合 $\mathrm{Var}(s)=nI(\theta)$，即得。

MLE 在正则条件下渐近达到该下界（渐近有效）。

### 6.4 指数族与充分统计量

指数族分布写成

$$
p_\theta(x)=h(x)\exp\Big(\eta(\theta)^{\top}T(x)-A(\eta)\Big),
\qquad
A(\eta)=\log\int h(x)e^{\eta^{\top}T(x)}\mathrm{d}x .
$$

性质（可由“对 $A$ 求导”证明）：

$$
\nabla_\eta A(\eta)=\mathbb{E}[T(X)],
\qquad
\nabla^2_\eta A(\eta)=\mathrm{Cov}(T(X))
\ \succeq 0,
$$

所以 $A$ 是凸函数、$T$ 是充分统计量（Fisher–Neyman 分解）。Categorical/softmax 是指数族：$T(x)=e_{y}$，$\eta=z$，$A(z)=\log\sum_k e^{z_k}$。**Softmax 的 log-partition 函数就是第 12 篇要反复使用的对象。**

---

## 7. 机器学习中的估计：经验风险与泛化

### 7.1 风险与经验风险

给定损失 $\ell(y,f_\theta(x))$：

$$
R(\theta)=\mathbb{E}_{(X,Y)\sim P}\big[\ell(Y,f_\theta(X))\big],
\qquad
\widehat R_n(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(y_i,f_\theta(x_i)).
$$

经验风险最小化（ERM）：$\widehat\theta=\arg\min_\theta \widehat R_n(\theta)$。概率建模中取 $\ell=-\log p_\theta(y\mid x)$，ERM 就退化为 MLE。

### 7.2 一致收敛与有限类泛化界

**Hoeffding 不等式.** 若 $\ell_i\in[a,b]$ 独立，则

$$
\mathbb{P}\Big(\big|\widehat R_n-R\big|\ge\varepsilon\Big)
\ \le\ 2\exp\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

（证明见本目录《2-霍夫丁不等式（Hoeffding's inequality)-zhihu.md》与《3-heoffding-proof-ds.md》。对 $|{\mathcal{H}}|$ 个候选函数的有限类，用联合界：）

$$
\mathbb{P}\Big(\exists h\in\mathcal{H}:\ |\widehat R_n(h)-R(h)|\ge\varepsilon\Big)
\le 2|\mathcal{H}|\exp\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

给定置信度 $\delta$，令右端为 $\delta$，解出

$$
\varepsilon=\sqrt{\frac{(b-a)^2\ln(2|\mathcal{H}|/\delta)}{2n}},
$$

即一致收敛速率 $O\big(\sqrt{\ln|\mathcal{H}|/n}\big)$。一般函数类的推广是 VC 维 / Rademacher 复杂度界。

### 7.3 预测器的偏差–方差分解

设 $Y=f^*(X)+\epsilon$，$\mathbb{E}[\epsilon]=0$，$\mathrm{Var}(\epsilon)=\sigma^2$，$\widehat f$ 由训练集学得。则在点 $x$：

$$
\mathbb{E}\big[(Y-\widehat f(X))^2\mid X=x\big]
=\sigma^2
+\Big(\mathbb{E}[\widehat f(x)]-f^*(x)\Big)^2
+\mathrm{Var}\big(\widehat f(x)\big).
$$

**证明.** 在 $x$ 处条件展开 $(Y-\widehat f)^2=\big(\epsilon+(f^*-\widehat f)\big)^2$，交叉项用 $\epsilon$ 零均值与独立性消去，再把 $f^*-\widehat f=(f^*-\mathbb{E}\widehat f)+(\mathbb{E}\widehat f-\widehat f)$ 平方展开，交叉项为零均值。$\square$

### 7.4 过参数化时代的注记

经典图景要求模型容量随 $n$ 增长而受控；现代大模型在插值阈值之后测试误差反而再降（双下降）。本系列不展开该话题，只需记住：**AlphaProof 的泛化依赖“搜索给出的已验证数据 + 大规模课程”，而不是经典 ERM 容量控制**（《6-alpha-proof-mcts-rl.md》§8、《9-mainrl-ttrl-llm-value-head.md》§1.3）。

---

## 8. 回到 AlphaProof：损失的估计论解释

- **策略损失**：对成功轨迹做交叉熵，是“搜索改进策略在成功动作上的条件分布”的 MLE；数据由当前策略驱动，近似 on-policy，因此无需显式信任域（《6-alpha-proof-mcts-rl.md》§6.1–6.2）。
- **价值损失**：64 个 bin 上的分类 MLE，目标是搜索实际回报 $G$。MLE 的最优解是经验回报分布，条件期望即 $\widehat V(s)\to\mathbb{E}[G\mid s]=-d(s)$（§3.2 的梯度条件和 §1.3 的条件期望最优性）。
- **AND 节点取 min**：这不是标准 CE 的最优解，而是环境对回报目标的重新定义（最难分支决定价值），带来子目标难度均衡的激励（《6-alpha-proof-mcts-rl.md》§1.2）。
- **$\lambda=10^{-3}$ 的价值权重**：在损失层对两个估计问题的相对尺度做校准；两个头共享编码器时，梯度是两路 VJP 的和（《10-grad-backprop-optimizer-gpu.md》§1.5、§4.4）。
- **蒙特卡洛回报而非 bootstrap**：价值目标无自举偏差，方差随证明长度增长——对应估计论中的“无偏但高方差估计量”，与 TD 的“有偏但低方差”形成取舍（备份 `12-rl-math.md` §4）。

---

## 9. 小结

- 条件期望是 $L^2$ 最优预测器，价值函数就是它的实例；
- MLE 最大化对数似然；对 categorical 模型其解是经验频率，对 softmax 参数化其梯度是“预测减 one-hot”；
- 最小化交叉熵等价于最小化 KL，MLE 是最小 KL 的样本版本；
- MSE 分解为偏差平方加方差；Fisher 信息与 Cramér–Rao 给出无偏估计的方差下界，MLE 渐近有效；
- 经验风险最小化把统计估计转成优化问题——下一批文档（12–15）就从这里进入函数形式（Softmax/MLP）、优化算法（梯度下降/Adam）与自动微分（反向传播）的完整公式。
