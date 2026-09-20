# 15 数学工具箱：线性代数、矩阵微积分、概率与凸性

参考：本系列《11-probability-mle-estimation.md》（概率与估计）、《12-softmax-logistic-mlp.md》（函数与导数）、《13-gradient-descent-and-optimizers.md》（优化引理）、《14-backprop-and-autodiff.md》（VJP 规则）、《10-grad-backprop-optimizer-gpu.md》（Transformer 逐层公式）。

定位：把前几篇反复使用但未系统列出的数学事实集中在一处：矩阵微积分的约定与恒等式、概率与信息论的定义与不等式、凸性、泰勒展开、浮点与复杂度。查漏补缺用；遇到不认识的符号先来这里。

---

## 1. 线性代数速查

### 1.1 内积、范数与正定性

$$
\langle x,y\rangle=x^{\top}y=\sum_i x_i y_i,
\qquad
\|x\|_2=\sqrt{\langle x,x\rangle},
\qquad
\|x\|_1=\sum_i|x_i|,
\qquad
\|x\|_\infty=\max_i|x_i|,
$$

$$
\|A\|_F=\sqrt{\sum_{ij}A_{ij}^2},
\qquad
\|A\|_2=\sigma_{\max}(A).
$$

对称矩阵 $A\in\mathbb{R}^{d\times d}$：

- 半正定 $A\succeq0$：$x^{\top}Ax\ge0$ 对所有 $x$；
- 正定 $A\succ0$：$x^{\top}Ax\gt0$ 对所有 $x\ne0$；
- 特征分解 $A=U\Lambda U^{\top}$（正交 $U$，实特征值 $\lambda_i$），正定当且仅当所有 $\lambda_i\gt0$。

### 1.2 奇异值分解与低秩

任意 $A\in\mathbb{R}^{m\times n}$：

$$
A=U\Sigma V^{\top},
\qquad
\Sigma=\mathrm{diag}(\sigma_1\ge\dots\ge\sigma_{\min(m,n)}\ge0),
$$

- 秩 $r=\#\{\sigma_i\gt0\}$；Eckart–Young 定理：秩 $r$ 的最优均方逼近由前 $r$ 个奇异值给出；
- 条件数 $\kappa(A)=\sigma_{\max}/\sigma_{\min}$（可逆方阵时等于 $\lambda$ 条件数）；
- 伪逆 $A^{+}=V\Sigma^{+}U^{\top}$；
- **LoRA** 用低秩乘积 $W=W_0+BA$（$B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$）近似增量，见《9-mainrl-ttrl-llm-value-head.md》§2.0。

### 1.3 正交矩阵与旋转

$R^{\top}R=RR^{\top}=I$ 时称 $R$ 正交。正交变换保内积、保 $\ell_2$ 范数，其反向（转置）就是逆。RoPE 的位置编码正是一族按位置旋转的二维旋转矩阵：$R_m^{\top}R_n=R_{n-m}$，因此注意力分数只依赖相对位置（备份 `10-transformers.md` §6）。

---

## 2. 矩阵微积分

### 2.1 布局约定

存在两套常用约定（分子布局与分母布局），混用是公式出错的常见原因。本系列统一采用**分母布局**的结果形式，即：

- 标量对向量 $\partial f/\partial x$ 是列向量（与 $x$ 同形）；
- 标量对矩阵 $\partial f/\partial X$ 与 $X$ 同形；
- 对 $f:\mathbb{R}^{d}\to\mathbb{R}^{k}$，Jacobian 取 $k\times d$（每行一个输出分量）。

### 2.2 微分法（推荐）

**核心技巧**：把标量函数的微分写成内积形式，再对照定义读出梯度。

$$
\mathrm{d}f=\sum_{ij}\frac{\partial f}{\partial X_{ij}}\mathrm{d}X_{ij}
=\mathrm{tr}\Big(\Big(\frac{\partial f}{\partial X}\Big)^{\top}\mathrm{d}X\Big).
$$

因此只要把 $\mathrm{d}f$ 整理成 $\mathrm{tr}(A^{\top}\mathrm{d}X)$，就有 $\partial f/\partial X=A$。

### 2.3 常用恒等式（附推导）

- $\dfrac{\partial(a^{\top}x)}{\partial x}=a$：$\mathrm{d}(a^{\top}x)=a^{\top}\mathrm{d}x$。

- $\dfrac{\partial(x^{\top}Ax)}{\partial x}=(A+A^{\top})x$：$\mathrm{d}(x^{\top}Ax)=(\mathrm{d}x)^{\top}Ax+x^{\top}A\,\mathrm{d}x=x^{\top}(A^{\top}+A)\mathrm{d}x$。

- $\dfrac{\partial\,\mathrm{tr}(AB)}{\partial A}=B^{\top}$：$\mathrm{d}\,\mathrm{tr}(AB)=\mathrm{tr}(\mathrm{d}A\,B)=\mathrm{tr}(B\,\mathrm{d}A)$。

- $\dfrac{\partial\,\mathrm{tr}(A^{\top}B)}{\partial A}=B$：$\mathrm{d}\,\mathrm{tr}(A^{\top}B)=\mathrm{tr}((\mathrm{d}A)^{\top}B)=\mathrm{tr}(B^{\top}\mathrm{d}A)$。

- $\dfrac{\partial\log\det A}{\partial A}=A^{-\top}$（$A$ 可逆）：由 $\mathrm{d}\log\det A=\mathrm{tr}(A^{-1}\mathrm{d}A)$ 与上式。

- $\dfrac{\partial A^{-1}}{\partial t}=-A^{-1}\dfrac{\partial A}{\partial t}A^{-1}$（对参数 $t$）：微分 $AA^{-1}=I$ 得。

- **链式法则**：$f(X)=g(h(X))$ 时按标量逐分量处理，或写成 VJP：$\bar X=(\partial h/\partial X)^{\top}\bar h$。

### 2.4 与反向传播的联系

$Y=XW$ 时 $\bar X=\bar YW^{\top}$、$\bar W=X^{\top}\bar Y$ 就是上表中 $\mathrm{tr}$ 恒等式的直接结果（《14-backprop-and-autodiff.md》§4.2）。**矩阵微积分的全部工作就是把“微分—转置—迹”循环练熟。**

---

## 3. 概率与信息论工具箱

### 3.1 常用分布

- Bernoulli：$\mathbb{P}(X=1)=\theta$，$\mathbb{E}X=\theta$，$\mathrm{Var}=\theta(1-\theta)$；
- Categorical：$\mathbb{P}(Y=k)=p_k$，$\mathbb{E}[e_Y]=p$，$\mathrm{Cov}(e_Y)=\mathrm{diag}(p)-pp^{\top}$；
- 均匀：$\mathbb{E}X=(a+b)/2$，$\mathrm{Var}=(b-a)^2/12$；
- 正态 $\mathcal{N}(\mu,\sigma^2)$：密度 $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$，熵 $\frac{1}{2}\log(2\pi e\sigma^2)$；
- 多维正态：$\mathbb{E}[(X-\mu)(X-\mu)^{\top}]=\Sigma$，二次型的期望 $\mathbb{E}[X^{\top}AX]=\mathrm{tr}(A\Sigma)+\mu^{\top}A\mu$。

### 3.2 条件期望与 Bayes

- 塔性质：$\mathbb{E}[\mathbb{E}[X\mid Y]]=\mathbb{E}[X]$；
- 条件方差分解：$\mathrm{Var}(X)=\mathbb{E}[\mathrm{Var}(X\mid Y)]+\mathrm{Var}(\mathbb{E}[X\mid Y])$；
- Bayes 公式：$p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}$；
- 条件期望是 $L^2$ 最优预测器（证明见《11-probability-mle-estimation.md》§1.3）。

### 3.3 熵、KL、交叉熵、互信息

$$
H(p)=-\sum_x p(x)\log p(x),
\qquad
H(p,q)=-\sum_x p(x)\log q(x),
$$

$$
D(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)},
\qquad
I(X;Y)=D\big(p(x,y)\,\|\,p(x)p(y)\big).
$$

恒等式与不等式：

- $H(p,q)=H(p)+D(p\,\|\,q)$；
- $D(p\,\|\,q)\ge0$（Jensen，证明见《11-probability-mle-estimation.md》§4.2）；
- $I(X;Y)=H(X)-H(X\mid Y)\ge0$；
- 数据处理不等式：若 $X\to Y\to Z$ 是马尔可夫链，则 $I(X;Z)\le I(X;Y)$（叙述）。

### 3.4 Jensen 不等式

$\phi$ 凸、$X$ 可积，则

$$
\phi\big(\mathbb{E}[X]\big)\le\mathbb{E}\big[\phi(X)\big].
$$

**证明要点.** 取 $\phi$ 在 $\mathbb{E}X$ 处的支撑超平面 $\phi(x)\ge\phi(\mathbb{E}X)+\phi'(\mathbb{E}X)(x-\mathbb{E}X)$，两边取期望。$\square$

它是几乎所有信息论不等式（KL 非负、EM、变分界）的源头。

### 3.5 指数族

$$
p_\theta(x)=h(x)\exp\big(\eta(\theta)^{\top}T(x)-A(\eta)\big),
\qquad
A(\eta)=\log\int h(x)e^{\eta^{\top}T(x)}\mathrm{d}x .
$$

$$
\nabla A(\eta)=\mathbb{E}[T(X)],
\qquad
\nabla^2A(\eta)=\mathrm{Cov}(T(X))\succeq0 .
$$

Softmax/categorical 是 $T=e_y$、$\eta=z$、$A(z)=\mathrm{LSE}(z)$ 的特例（《12-softmax-logistic-mlp.md》§2.3）。

---

## 4. 凸性工具箱

### 4.1 定义与判定

- 凸集：$\theta,\theta'\in C\Rightarrow t\theta+(1-t)\theta'\in C$；
- 凸函数：$f(t\theta+(1-t)\theta')\le tf(\theta)+(1-t)f(\theta')$；
- 一阶判定（可微）：$f(\theta')\ge f(\theta)+\langle\nabla f(\theta),\theta'-\theta\rangle$；
- 二阶判定（二阶可微）：$\nabla^2f(\theta)\succeq0$。

**证明思路（二阶判定）.** 沿方向 $v$ 的一元函数 $g(t)=f(\theta+tv)$ 满足 $g''(t)=v^{\top}\nabla^2f(\theta+tv)v\ge0$，故 $g$ 凸；对任意两点取 $v=\theta'-\theta$ 得结论。$\square$

### 4.2 保凸运算

- 非负加权和 $f=\sum_i w_if_i$（$w_i\ge0$）凸；
- 仿射复合 $f(Ax+b)$ 凸（$f$ 凸）；
- 逐点最大值 $\max_i f_i$ 凸；
- 部分最小化 $g(x)=\inf_y f(x,y)$ 凸（$f$ 联合凸）。

这些规则解释了为什么“ReLU、max、hinge、CE”都是凸的积木。

### 4.3 次梯度

凸函数不可微处用次梯度集合

$$
\partial f(\theta)=\{g:\ f(\theta')\ge f(\theta)+\langle g,\theta'-\theta\rangle,\ \forall\theta'\}.
$$

例：$f(x)=|x|$ 在 0 处的次梯度是 $[-1,1]$；ReLU 在 0 处是 $[0,1]$。次梯度法把梯度下降推广到非光滑情形（收敛率 $O(1/\sqrt{k})$）。

### 4.4 强凸的后果

$\mu$-强凸、$L$-光滑的 $f$：

$$
\frac{\mu}{2}\|\theta-\theta^\star\|^2\le f(\theta)-f^\star\le\frac{L}{2}\|\theta-\theta^\star\|^2,
$$

$$
\|\nabla f(\theta)\|^2\ge2\mu\big(f(\theta)-f^\star\big)
\quad(\text{Polyak--Lojasiewicz 不等式}),
$$

它是线性收敛证明的关键（《13-gradient-descent-and-optimizers.md》§2.3）。

---

## 5. 微积分与泰勒展开

### 5.1 中值定理与积分余项

一元 $f\in C^1$：$f(b)-f(a)=\int_a^b f'(t)\mathrm{d}t$；$f\in C^2$ 时

$$
f(b)=f(a)+f'(a)(b-a)+\int_0^1(1-t)f''\big(a+t(b-a)\big)(b-a)^2\mathrm{d}t .
$$

**多元形式**（本系列反复使用）：

$$
f(\theta')=f(\theta)+\langle\nabla f(\theta),\theta'-\theta\rangle
+\int_0^1(1-t)\,(\theta'-\theta)^{\top}\nabla^2f\big(\theta+t(\theta'-\theta)\big)(\theta'-\theta)\,\mathrm{d}t .
$$

由 $\nabla^2f\preceq LI$ 立即得到下降引理中的二次上界（《13-gradient-descent-and-optimizers.md》§2.2）。

### 5.2 常用初等展开

$$
e^x=1+x+\frac{x^2}{2}+O(x^3),
\qquad
\log(1+x)=x-\frac{x^2}{2}+O(x^3),
\qquad
(1+x)^p=1+px+O(x^2).
$$

### 5.3 梯度与 Hessian 的几何

- $\nabla f(\theta)$ 指向最速上升方向，$-\nabla f$ 是最速下降；
- $\nabla^2f$ 的特征值决定曲率：正特征值方向是“碗”，负特征值方向是“鞍”；
- 条件数 $\kappa$ 大时等高线狭长，一阶方法收敛慢（《13-gradient-descent-and-optimizers.md》§2.3）。

---

## 6. 复杂度、浮点与工程常数

### 6.1 复杂度记号

$O(\cdot)$、$\Omega(\cdot)$、$\Theta(\cdot)$ 描述 $n\to\infty$ 的渐近阶。本系列的例子：

- 稠密矩阵乘 $n\times n$ 乘 $n\times n$：算术复杂度 $O(n^3)$；
- 注意力分数矩阵：$O(L^2d)$；
- Adam 单步：$O(\#\text{params})$；
- 全网络一次前向：$2N$ FLOPs/token，训练总计约 $6N$（《10-grad-backprop-optimizer-gpu.md》§6.1）。

### 6.2 浮点格式

- fp32：1 符号 + 8 指数 + 23 尾数，机器精度 $\approx6\times10^{-8}$；
- fp16：5 指数 + 10 尾数，动态范围小，需要 loss scaling；
- bf16：8 指数 + 7 尾数，动态范围与 fp32 相同、精度低，训练中常用“bf16 计算 + fp32 累加/主权重”；
- 舍入误差的积累使“先算概率再取 log”这类操作在大词表下不稳，必须用 log-softmax / LSE（《12-softmax-logistic-mlp.md》§7）。

### 6.3 数值稳定的两个模式

$$
\mathrm{LSE}(z)=m+\log\sum_j e^{z_j-m},
\qquad m=\max_j z_j,
$$

$$
\log\mathrm{softmax}(z)_k=z_k-\mathrm{LSE}(z).
$$

在线版本（分块递推）见《10-grad-backprop-optimizer-gpu.md》§6.3。

---

## 7. 工具与各篇的对应关系

- 条件期望与 MLE：价值函数、两个损失的概率解释（《11-probability-mle-estimation.md》）；
- Softmax Jacobian、$p-e_y$、LSE：策略/价值头与所有输出层（《12-softmax-logistic-mlp.md》）；
- 光滑/强凸/下降引理、Adam 偏差修正：优化器选择与学习率设计（《13-gradient-descent-and-optimizers.md》）；
- 链式法则、VJP 规则库、重计算：全网络梯度计算（《14-backprop-and-autodiff.md》）与 Transformer 逐层反传（《10-grad-backprop-optimizer-gpu.md》§4）；
- 矩阵微积分恒等式：手写反向与检查形状（本文 §2）；
- KL/Jensen/指数族：正则化、信任域、PPO/自然梯度类方法的理论背景（《13-gradient-descent-and-optimizers.md》§6–7）。
