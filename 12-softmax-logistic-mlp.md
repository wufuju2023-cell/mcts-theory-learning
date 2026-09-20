# 12 Softmax、Logistic 回归与 MLP：从线性模型到 Transformer 组件

参考：本系列《6-alpha-proof-mcts-rl.md》§7、《10-grad-backprop-optimizer-gpu.md》§1、§4，以及 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/10-transformers.md`（前向公式）与 `02-proof-network.md`（网络结构）。概率与 MLE 的铺垫见《11-probability-mle-estimation.md》。

定位：把 AlphaProof 里反复出现的三个函数——softmax（策略与价值输出层）、logistic/线性层（所有投影）、MLP（价值头与 FFN）——从最基础的形式推导到反向传播所需的全部 Jacobian。

---

## 1. 从线性模型到概率输出

### 1.1 二分类与 sigmoid

给定特征 $x\in\mathbb{R}^{d}$，线性打分

$$
z=w^{\top}x+b\in\mathbb{R}.
$$

把打分压到 $(0,1)$ 得到概率：

$$
\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
p(y=1\mid x)=\sigma(z),
\qquad
p(y=0\mid x)=1-\sigma(z).
$$

**性质.** $\sigma(-z)=1-\sigma(z)$；导数

$$
\sigma'(z)=\frac{e^{-z}}{(1+e^{-z})^2}=\sigma(z)\big(1-\sigma(z)\big);
$$

对数几率（log-odds）是线性的：

$$
\log\frac{p(y=1\mid x)}{p(y=0\mid x)}=z=w^{\top}x+b .
$$

### 1.2 二分类的负对数似然

对样本 $(x_i,y_i)$，$y_i\in\{0,1\}$，负对数似然

$$
\ell_i=-\log p(y_i\mid x_i)
=-\Big[y_i\log\sigma(z_i)+(1-y_i)\log\big(1-\sigma(z_i)\big)\Big].
$$

对 $z_i$ 求导（用 sigmoid 的导数）：

$$
\frac{\partial \ell_i}{\partial z_i}
=-\Big[\frac{y_i}{\sigma(z_i)}-\frac{1-y_i}{1-\sigma(z_i)}\Big]\sigma(z_i)\big(1-\sigma(z_i)\big)
=\sigma(z_i)-y_i .
$$

对 $w,b$ 的梯度由链式法则给出：

$$
\nabla_w \ell_i=(\sigma(z_i)-y_i)\,x_i,
\qquad
\frac{\partial \ell_i}{\partial b}=\sigma(z_i)-y_i .
$$

这是“预测减标签”模式的最简版本；softmax 只是它的多类推广。

---

## 2. Softmax 的定义与性质

### 2.1 定义

对 logits $z\in\mathbb{R}^{K}$：

$$
\mathrm{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},
\qquad k=1,\dots,K .
$$

**性质.**

- 输出在单纯形内：$p_k\gt 0$，$\sum_k p_k=1$；
- 平移不变：$\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$；
- 单调：$z_k\gt z_j\Rightarrow p_k\gt p_j$；
- 极限：$\tau\to0$ 时 $\mathrm{softmax}(z/\tau)\to$ one-hot(argmax)；$\tau\to\infty$ 时趋于均匀（温度行为；先验温度恒等式见《6-alpha-proof-mcts-rl.md》§5.1）；
- 与 log-sum-exp 的关系：

$$
\log \mathrm{softmax}(z)_k=z_k-\mathrm{LSE}(z),
\qquad
\mathrm{LSE}(z)=\log\sum_j e^{z_j}.
$$

### 2.2 Jacobian（完整推导）

记 $p=\mathrm{softmax}(z)$。分两种情形：

$$
\frac{\partial p_k}{\partial z_k}
=\frac{e^{z_k}\sum_j e^{z_j}-e^{z_k}e^{z_k}}{\big(\sum_j e^{z_j}\big)^2}
=p_k-p_k^2=p_k(1-p_k),
$$

$$
\frac{\partial p_k}{\partial z_j}
=-\frac{e^{z_k}e^{z_j}}{\big(\sum_j e^{z_j}\big)^2}
=-p_k p_j
\qquad (j\ne k).
$$

合并：

$$
\boxed{\;
\frac{\partial p}{\partial z}
=\mathrm{diag}(p)-p\,p^{\top}
\;}
$$

该矩阵对称、半正定、行和为 0（因为 $\sum_k p_k=1$ 是恒等式）。把它作用到伴随向量 $\bar p$：

$$
\bar z=\big(\mathrm{diag}(p)-pp^{\top}\big)\bar p
=p\odot\big(\bar p-(\mathbf{1}^{\top}\bar p)\mathbf{1}\big),
$$

逐分量即

$$
\bar z_k=p_k\big(\bar p_k-\textstyle\sum_j p_j\bar p_j\big).
$$

（第 10 篇 §4.1 引理 1 给出按行处理的注意力版本；这里是一般的向量版本。）

### 2.3 作为指数族与 log-partition

Categorical 分布属于指数族：

$$
p(z)_k=\exp\Big(z_k-\underbrace{\log\sum_j e^{z_j}}_{A(z)}\Big),
\qquad
A(z)=\mathrm{LSE}(z).
$$

$A$ 是凸函数（其 Hessian 正是 $\mathrm{diag}(p)-pp^{\top}\succeq0$），并且

$$
\nabla A(z)=\mathbb{E}_{p}[e_y]=p,
\qquad
\nabla^2 A(z)=\mathrm{Cov}(e_y)=\mathrm{diag}(p)-pp^{\top}.
$$

这给出 softmax 的两个“免费”结论：Hessian 半正定、以及 $\nabla\,\mathrm{LSE}=p$。

---

## 3. 多项 logistic 回归与交叉熵

### 3.1 模型

$$
p(y=k\mid x)=\mathrm{softmax}(Wx+b)_k,
\qquad
z=Wx+b\in\mathbb{R}^{K}.
$$

对数似然与损失：

$$
\ell=\sum_{i=1}^{n}\log p(y_i\mid x_i),
\qquad
\mathcal{L}=-\ell .
$$

### 3.2 梯度（完整推导）

记 one-hot 目标 $e_{y}\in\{0,1\}^{K}$，$p_i=\mathrm{softmax}(z_i)$。对单个样本：

$$
\frac{\partial \mathcal{L}_i}{\partial z_i}
=p_i-e_{y_i},
$$

（直接对 $-\log \mathrm{softmax}(z)_{y}=-(z_y-\mathrm{LSE}(z))$ 求导，或用 §2.2 的 Jacobian 与 $-\nabla_z\log p_y=-(e_y-p)=p-e_y$。）于是

$$
\nabla_W\mathcal{L}_i=(p_i-e_{y_i})\,x_i^{\top},
\qquad
\nabla_b\mathcal{L}_i=p_i-e_{y_i}.
$$

### 3.3 凸性

单个样本的 Hessian（对 $z$）是 $\mathrm{diag}(p)-pp^{\top}\succeq0$；对 $W$ 的 Hessian 是 $x x^{\top}\otimes(\mathrm{diag}(p)-pp^{\top})\succeq0$。因此**多项 logistic 回归的负对数似然是凸函数**，全局最优的一阶条件恰是 §3.2 的梯度为零，即第 11 篇 §3.2 的经验频率结论。

### 3.4 在 AlphaProof 中的三个实例

- 价值头：$K=64$ 个 bin，$z_v=\mathrm{MLP}(\bar h)$，$\mathcal{L}_{\text{val}}=\mathrm{CE}(p_v,y)$，权重 $10^{-3}$（Supplementary Table 6；《10-grad-backprop-optimizer-gpu.md》§1.4）；
- 策略头：$K=|V|$（词表），逐 token $\mathcal{L}_{\text{pol}}=-\log p_\theta(a_j\mid a_{<j},s)$；
- 搜索先验：对策略分布取 $1/\tau$ 次幂即对 logits 除以 $\tau$（《6-alpha-proof-mcts-rl.md》§5.1）。

---

## 4. MLP（多层感知机）

### 4.1 定义

一个 $L$ 层 MLP 是交替的仿射变换与逐元素非线性：

$$
h^{(0)}=x,
\qquad
h^{(\ell)}=\phi\big(W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}\big),
\quad \ell=1,\dots,L,
\qquad
f_\theta(x)=W^{(L+1)}h^{(L)}+b^{(L+1)} .
$$

逐元素非线性 $\phi$ 是必要的：若去掉，复合仍是仿射，深度没有意义。

### 4.2 常用激活函数与导数

- **ReLU**：$\phi(x)=\max(0,x)$，$\phi'(x)=\mathbf{1}[x\gt0]$（次梯度在 0 处取 $[0,1]$）；
- **LeakyReLU**：$\phi(x)=\max(\alpha x,x)$；
- **tanh**：$\phi'=1-\phi^2$；
- **GELU**（平滑版 ReLU）：$\phi(x)=x\,\Phi(x)$，$\Phi$ 是标准正态分布函数，$\phi'(x)=\Phi(x)+x\,\varphi(x)$；
- **SiLU/Swish**：$\phi(x)=x\,\sigma(x)$，导数

$$
\phi'(x)=\sigma(x)+x\,\sigma(x)\big(1-\sigma(x)\big)
=\sigma(x)\big(1+x\,(1-\sigma(x))\big).
$$

（Transformer 的 FFN 使用 SiLU 构成 SwiGLU，见 §6。）

### 4.3 表示能力：万能逼近定理（叙述）

**定理（Cybenko/Hornik 型）.** 设 $\sigma$ 是非常数、有界、单调连续的函数（或 ReLU）。对紧集 $K\subset\mathbb{R}^{d}$ 上的任意连续函数 $f$ 与任意 $\varepsilon\gt0$，存在单隐层网络

$$
g(x)=\sum_{j=1}^{m} c_j\,\phi\big(a_j^{\top}x+b_j\big)
$$

使 $\sup_{x\in K}|f(x)-g(x)|\lt\varepsilon$。

直觉：隐层单元可以构造“阶跃/凸起”基底，足够多的小块以任意精度拼出连续函数。深度版本更省参数（某些函数需要指数宽的浅网络与多项式宽的深网络），这也是 Transformer 采用深层堆叠的动机之一。定理只保证存在性，不保证优化能找到，也不保证泛化。

### 4.4 一般 MLP 的前向与反向（递推）

前向保存每层的输入（激活重计算时另说）：

$$
a^{(\ell)}=W^{(\ell)}h^{(\ell-1)}+b^{(\ell)},
\qquad
h^{(\ell)}=\phi\big(a^{(\ell)}\big).
$$

给定最后一层的伴随 $\delta^{(L)}=\partial\mathcal{L}/\partial h^{(L)}$（由损失函数给出），反向递推：

$$
\delta_a^{(\ell)}=\delta^{(\ell)}\odot\phi'\big(a^{(\ell)}\big),
$$

$$
\nabla_{W^{(\ell)}}\mathcal{L}=\delta_a^{(\ell)}\,\big(h^{(\ell-1)}\big)^{\top},
\qquad
\nabla_{b^{(\ell)}}\mathcal{L}=\delta_a^{(\ell)},
\qquad
\delta^{(\ell-1)}=\big(W^{(\ell)}\big)^{\top}\delta_a^{(\ell)} .
$$

这就是第 14 篇“反向模式自动微分”的特例；矩阵情形的形状约定在那里给出。

### 4.5 两层 MLP + 交叉熵的完整反向（手推示例）

取 $L=1$，输出 softmax 交叉熵：

$$
z=W^{(2)}h^{(1)}+b^{(2)},
\qquad
p=\mathrm{softmax}(z),
\qquad
\mathcal{L}=-\log p_y .
$$

**反向.** 由 §3.2，$\delta_z=p-e_y$。于是

$$
\nabla_{W^{(2)}}\mathcal{L}=\delta_z\,h^{(1)\top},
\qquad
\nabla_{b^{(2)}}\mathcal{L}=\delta_z,
$$

$$
\delta_{h^{(1)}}=W^{(2)\top}\delta_z,
\qquad
\delta_{a^{(1)}}=\delta_{h^{(1)}}\odot\phi'\big(a^{(1)}\big),
$$

$$
\nabla_{W^{(1)}}\mathcal{L}=\delta_{a^{(1)}}x^{\top},
\qquad
\nabla_{b^{(1)}}\mathcal{L}=\delta_{a^{(1)}} .
$$

全部是矩阵乘或逐元素乘，没有近似；这就是“反向传播 = 链式法则 + 动态规划”。

---

## 5. 一个矩阵参数的梯度与形状检查

若 $Y=XW$（$X$ 是 $T\times d$，$W$ 是 $d\times k$），则

$$
\nabla_W\mathcal{L}=X^{\top}\,\delta_Y,
\qquad
\delta_X=\delta_Y\,W^{\top}.
$$

形状检查：$\delta_Y$ 是 $T\times k$，$X^{\top}$ 是 $d\times T$，乘积 $d\times k$ 与 $W$ 同形；第二式 $T\times k$ 乘 $k\times d$ 得 $T\times d$，与 $X$ 同形。**任何“形状对不上”的反向公式都是错的**——这是实现自动微分之外手写反向时的第一道自检。

---

## 6. Transformer 中的 MLP 与输出头

### 6.1 SwiGLU FFN

备份 `10-transformers.md` §3 的写法（$W_1,W_g\in\mathbb{R}^{d\times d_{\text{ff}}}$，$W_2\in\mathbb{R}^{d_{\text{ff}}\times d}$）：

$$
\mathrm{FFN}(z)=\Big(\big(zW_1\big)\odot\mathrm{SiLU}\big(zW_g\big)\Big)W_2 .
$$

记 $P=zW_1$，$G=zW_g$，$C=P\odot\mathrm{SiLU}(G)$，$Y=CW_2$，则反向（第 10 篇 §4.1 引理 5）：

$$
\delta_C=\delta_YW_2^{\top},
\qquad
\delta_P=\delta_C\odot\mathrm{SiLU}(G),
\qquad
\delta_G=\delta_C\odot P\odot\mathrm{SiLU}'(G),
$$

$$
\nabla_{W_2}=C^{\top}\delta_Y,
\qquad
\nabla_{W_1}=z^{\top}\delta_P,
\qquad
\nabla_{W_g}=z^{\top}\delta_G,
\qquad
\delta_z=\delta_PW_1^{\top}+\delta_GW_g^{\top}.
$$

### 6.2 价值头与策略头

- 价值头是一个小 MLP 加 softmax（64 类），反向见《10-grad-backprop-optimizer-gpu.md》§4.2；
- 策略头是解码器最后一个线性层（与词嵌入绑定），反向见同上 §4.3；
- 两个头共享编码器，梯度在编码器处相加：$\nabla_{\theta_{\text{enc}}}\mathcal{L}=\nabla_{\theta_{\text{enc}}}\mathcal{L}_{\text{pol}}+10^{-3}\nabla_{\theta_{\text{enc}}}\mathcal{L}_{\text{val}}$。

---

## 7. 数值稳定与工程要点

- **log-sum-exp 稳定化**：

$$
\mathrm{LSE}(z)=m+\log\sum_j e^{z_j-m},
\qquad m=\max_j z_j,
$$

因为 $z_j-m\le0$，指数不会上溢；$\log\mathrm{softmax}(z)_k=z_k-\mathrm{LSE}(z)$。

- **在线 LSE**（分块/流式，用于 FlashAttention 与融合交叉熵）：维护 $(m,\ell)$ 三元组并按

$$
m_{\text{new}}=\max(m,m_2),
\qquad
\ell_{\text{new}}=\ell e^{\,m-m_{\text{new}}}+\ell_2 e^{\,m_2-m_{\text{new}}}
$$

递推（《10-grad-backprop-optimizer-gpu.md》§6.3）。

- **融合 softmax-CE**：不物化 $p$，直接在块内算 logits、LSE 与损失，反向输出 $p-e_y$（同上 §6.4）；
- **温度**：推理采样用 $p_\tau=\mathrm{softmax}(z/\tau)$；精度上温度不改变 argmax 顺序，只改变分布锐度；
- **数值精度**：bf16 下 softmax 的分母与 LSE 在 fp32 累加；小概率 class 的 log 值要用 log-softmax 而不是先算概率再取对数。

---

## 8. 小结

- sigmoid/softmax 是把线性打分变成概率的归一化算子，Jacobian 分别是 $\sigma(1-\sigma)$ 与 $\mathrm{diag}(p)-pp^{\top}$；
- 交叉熵对 logits 的梯度统一为“预测减 one-hot”：$p-e_y$；
- MLP 是仿射加逐元素非线性，反向是逐层矩阵乘加逐元素乘；
- Transformer 的 FFN（SwiGLU）、价值头（小 MLP）、策略头（绑定解嵌）都是本文公式的直接实例；
- 下一批文档：优化器与梯度下降（13）、反向传播与自动微分的系统化（14）、以及配套数学工具（15）。
