# 14 反向传播与自动微分：从链式法则到 VJP 规则库

参考：本系列《10-grad-backprop-optimizer-gpu.md》§2、§4（Transformer 逐层反传的完整公式）、《12-softmax-logistic-mlp.md》§4–6（MLP 前向与反向）、《11-probability-mle-estimation.md》（损失的概率含义）。本机旧笔记 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/10-transformers.md`（前向公式）。

定位：把“反向传播”从链式法则一步步推广为通用的反向模式自动微分（reverse-mode AD）：定义、复杂度、VJP 规则库、内存管理（重计算）、数值检验，以及与 JAX `jax.grad` 的对应。

---

## 1. 计算图与链式法则

### 1.1 计算图

一个可微程序可以写成有向无环图（DAG）：

- 节点：张量（输入、参数、中间量、输出）；
- 边：初等算子（加法、乘法、矩阵乘、逐元素函数、softmax、归一化……）。

设节点 $v$ 的输出为 $y$，其子节点为 $u_1,\dots,u_m$，局部映射 $y=f(u_1,\dots,u_m)$。

### 1.2 多元链式法则

若 $\mathcal{L}=\mathcal{L}(y_1,\dots,y_k)$ 且每个 $y_j$ 依赖 $x$，则

$$
\frac{\partial\mathcal{L}}{\partial x}
=\sum_{j=1}^{k}\frac{\partial\mathcal{L}}{\partial y_j}\frac{\partial y_j}{\partial x},
$$

矩阵形式是 Jacobian 的乘积与求和。链式法则本身没有规定计算顺序：

- **从左往右**（先乘 $\partial\mathcal{L}/\partial y$ 与 $\partial y/\partial x$）：反向模式；
- **从右往左**（先乘 $\partial y/\partial x$ 与 $\dot x$）：前向模式。

两种顺序数学上等价，代价可以相差几个数量级。

---

## 2. 前向模式与反向模式

### 2.1 前向模式（JVP）

给定输入扰动 $\dot x=\mathrm{d}x/\mathrm{d}t$，沿计算图正向传播方向导数：

$$
\dot y=\frac{\partial y}{\partial x}\,\dot x
\qquad(\text{Jacobian-vector product, JVP}).
$$

一次前向传播得到一个“输入方向”的导数；要得到完整 Jacobian 的 $d_{\text{in}}$ 列需要 $d_{\text{in}}$ 次。

### 2.2 反向模式（VJP）

给定输出侧的伴随 $\bar y=\partial\mathcal{L}/\partial y$，沿图反向传播：

$$
\bar x
=\Big(\frac{\partial y}{\partial x}\Big)^{\top}\bar y
\qquad(\text{vector-Jacobian product, VJP}).
$$

一次反向传播得到一个“输出方向”的导数；**当输出是标量（损失）时，一次反向就得到全部输入的梯度**。深度网络的参数以百万到十亿计、损失是标量，因此反向模式是唯一实用的选择。

### 2.3 复杂度定理（叙述）

对由初等算子组成的计算图，反向模式求标量损失的梯度，其算术运算量是前向的常数倍（每个初等算子约 2–3 倍），额外内存正比于保存的中间激活。对矩阵乘链 $Y_1=X W_1,\ Y_2=Y_1 W_2,\dots$，反向每层做两次同规模 GEMM；这就是《10-grad-backprop-optimizer-gpu.md》§6.1 的 $6N$ 系数的来源。

---

## 3. 反向模式 AD 的算法结构

### 3.1 伴随递推

对图中每个节点 $v$，维护伴随

$$
\bar v=\frac{\partial\mathcal{L}}{\partial v}.
$$

初始化 $\bar{\mathcal{L}}=1$。按拓扑逆序处理节点：若 $v$ 是 $w_1,\dots,w_m$ 的父节点（$v$ 的输出参与计算 $w_j$），则

$$
\bar v
=\sum_{j=1}^{m}
\Big(\frac{\partial w_j}{\partial v}\Big)^{\top}\bar w_j .
$$

实现上是“记录带（tape）”：前向时把算子与所需中间量（或用于重算的输入）记录下来；反向时逐条弹出算子，调用它的 VJP 规则并把结果累加到相应输入的伴随上。

### 3.2 内存与重计算（gradient checkpointing）

反向需要前向的中间值（例如注意力里的 $Q,K$、归一化的分母 $r$）。朴素做法保存全部激活，显存随深度线性增长。重计算（checkpointing）是分段保存：

- 只保存每 $k$ 层的边界激活；
- 反向到某段时，从边界重算该段激活，再算梯度。

**代价核算.** 设前向计算量为 $F$，保存所有激活的显存为 $M$。每 $k$ 层做一次检查点：显存降为约 $M/k$，前向计算量增为约 $F+F/k$（每个检查点段多算一次前向）。取 $k=\sqrt{n}$（$n$ 为层数）可得显存 $O(\sqrt{n})$ 与计算量 $(1+1/\sqrt{n})F$——这是 Transformer 训练（含 AlphaProof 的 18+24 层编码解码器）的标准做法（《10-grad-backprop-optimizer-gpu.md》§6.4）。

---

## 4. 常用算子的 VJP 规则库（含推导）

约定：$\mathcal{L}$ 是标量损失，$\bar Y=\partial\mathcal{L}/\partial Y$。

### 4.1 加法与逐元素运算

- $Y=X_1+X_2$：$\bar X_1=\bar X_2=\bar Y$；
- $Y=X_1\odot X_2$：$\bar X_1=\bar Y\odot X_2$，$\bar X_2=\bar Y\odot X_1$；
- $Y=f(X)$（逐元素）：$\bar X=\bar Y\odot f'(X)$；
- 广播的逆操作是“沿被广播的维度求和”：若 $Y=X+b$（$b$ 是行向量），则 $\bar b=\mathrm{colsum}(\bar Y)$。

### 4.2 矩阵乘法

$Y=XW$：

$$
\bar X=\bar Y\,W^{\top},
\qquad
\bar W=X^{\top}\bar Y .
$$

**推导.** $\mathrm{d}Y=\mathrm{d}X\,W+X\,\mathrm{d}W$，于是

$$
\mathrm{d}\mathcal{L}
=\mathrm{tr}\big(\bar Y^{\top}\mathrm{d}X\,W\big)+\mathrm{tr}\big(\bar Y^{\top}X\,\mathrm{d}W\big)
=\mathrm{tr}\big((\bar Y W^{\top})^{\top}\mathrm{d}X\big)+\mathrm{tr}\big((X^{\top}\bar Y)^{\top}\mathrm{d}W\big),
$$

对照 $\mathrm{d}\mathcal{L}=\mathrm{tr}(\bar X^{\top}\mathrm{d}X)+\mathrm{tr}(\bar W^{\top}\mathrm{d}W)$ 即得。批量情形把 batch 维求和（把 $X,Y$ 展平成二维即可）。

### 4.3 转置、reshape、切片、拼接

- $Y=X^{\top}$：$\bar X=\bar Y^{\top}$；
- reshape：$\bar X=\mathrm{reshape}(\bar Y)$（形状还原）；
- 切片 $Y=X_{[a:b]}$：$\bar X$ 在切片位置填 $\bar Y$，其余为零；
- 拼接 $Y=[X_1;X_2]$：$\bar X_1,\bar X_2$ 是相应的切片。

### 4.4 求和、均值、广播

- $Y=\sum_i X_i$：$\bar X_i=\bar Y$；
- $Y=\frac{1}{n}\sum_i X_i$：$\bar X_i=\bar Y/n$；
- $Y=X\mathbf{1}$（按行求和）：$\bar X=\bar Y\,\mathbf{1}^{\top}$。

### 4.5 除法与幂

$Y=X/Z$：$\bar X=\bar Y/Z$，$\bar Z=-\bar Y\odot X/Z^2$。$Y=X^p$：$\bar X=p\,\bar Y\odot X^{p-1}$（标量 $p$）。

### 4.6 exp、log 与 logsumexp

- $Y=\exp(X)$：$\bar X=\bar Y\odot Y$；
- $Y=\log(X)$：$\bar X=\bar Y\oslash X$；
- $Y=\mathrm{LSE}(X)=\log\sum_j e^{X_j}$（一行）：$\bar X_j=\bar Y\,p_j$，$p=\mathrm{softmax}(X)$——这正是 §4.7 的特例，也解释了“LSE 的梯度是 softmax”。

### 4.7 Softmax

$P=\mathrm{softmax}_{\text{row}}(Z)$，由《10-grad-backprop-optimizer-gpu.md》§4.1 引理 1：

$$
\bar Z=P\odot\Big(\bar P-\mathrm{rowsum}(\bar P\odot P)\,\mathbf{1}^{\top}\Big).
$$

### 4.8 交叉熵 + softmax（组合规则）

$P=\mathrm{softmax}(z)$，$\mathcal{L}=-\log P_y$：

$$
\bar z=P-e_y .
$$

推导：$\bar P_k=-\delta_{ky}/P_y$，代入 softmax 的 VJP 后化简（或用 $\mathcal{L}=-(z_y-\mathrm{LSE}(z))$ 直接求导）。**组合算子比分开两个算子更稳定、更省内存**——这是框架提供 `softmax_cross_entropy` 融合算子的原因。

### 4.9 归一化

RMSNorm（《10-grad-backprop-optimizer-gpu.md》§4.1 引理 3）：

$$
y=\gamma\odot\frac{x}{r},
\qquad
r=\sqrt{\tfrac{1}{d}\sum_j x_j^2+\varepsilon},
$$

$$
\bar x=\frac{\gamma\odot\bar y}{r}
-\frac{x}{d\,r^{3}}\big\langle x,\ \gamma\odot\bar y\big\rangle,
\qquad
\bar\gamma=\bar y\odot\frac{x}{r}.
$$

LayerNorm 多一项去均值：$y=\gamma\odot\frac{x-\mu}{\sigma}$，其 VJP 可由“减去行均值”与 RMSNorm 的规则组合得到（$\bar x\leftarrow\bar x-\mathrm{rowmean}(\bar x)$ 的投影性质）。

### 4.10 注意力（单头）

$S=QK^{\top}/\sqrt{d_k}$，$A=\mathrm{softmax}_{\text{row}}(S)$，$O=AV$：

$$
\bar V=A^{\top}\bar O,
\qquad
\bar A=\bar O V^{\top},
\qquad
\bar S=A\odot\big(\bar A-\mathrm{rowsum}(\bar A\odot A)\mathbf{1}^{\top}\big),
$$

$$
\bar Q=\frac{1}{\sqrt{d_k}}\bar S K,
\qquad
\bar K=\frac{1}{\sqrt{d_k}}\bar S^{\top}Q .
$$

（推导见《10-grad-backprop-optimizer-gpu.md》§4.1 引理 4。）Causal mask 的处理：被掩位置令 $A=0$，其伴随自动为 0，无需特殊分支。

### 4.11 最大值与条件选择

- $Y=\max(X_1,X_2)$：梯度只流向取到最大值的那个输入（并列时可取平均）；
- $Y=\mathrm{where}(c,X_1,X_2)$：$\bar X_1$ 与 $\bar X_2$ 只在选择位置非零；
- $Y=\mathrm{stop\_gradient}(X)$：$\bar X=0$（用于目标网络、EMA、采样动作等“不回传”的路径）。

### 4.12 gather / scatter

- 嵌入查表 $y=W[v]$：$\bar W[v]\mathrel{+}=\bar y$（散射累加，重复 token 需要原子加或分段规约）；
- 采样 $y\sim p(\cdot)$：采样操作不可微，梯度为零；用重参数化技巧（如 Gumbel-softmax）时按对应算子处理。

---

## 5. 反向传播 = 反向模式 AD 的特例

把神经网络写成

$$
h^{(\ell)}=\phi\big(W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}\big),
\qquad
\mathcal{L}=\ell\big(h^{(L)},y\big),
$$

应用 §3 的伴随递推与 §4 的规则，得到的就是教科书里的反向传播公式（《12-softmax-logistic-mlp.md》§4.4）：

$$
\delta^{(\ell)}=W^{(\ell+1)\top}\Big(\delta^{(\ell+1)}\odot\phi'\big(a^{(\ell+1)}\big)\Big),
\qquad
\nabla_{W^{(\ell)}}=\big(\delta^{(\ell)}\odot\phi'(a^{(\ell)})\big)h^{(\ell-1)\top}.
$$

**“反向传播”是反向模式 AD 在分层结构上的具体化；“自动微分”是同一件事在任意程序上的系统化实现。**

---

## 6. 三个完整实例

### 6.1 两层 MLP + 交叉熵

前向：

$$
a^{(1)}=W^{(1)}x+b^{(1)},
\quad
h^{(1)}=\mathrm{SiLU}\big(a^{(1)}\big),
\quad
z=W^{(2)}h^{(1)}+b^{(2)},
\quad
p=\mathrm{softmax}(z),
\quad
\mathcal{L}=-\log p_y .
$$

反向（按 §4 规则）：

$$
\bar z=p-e_y
\;\rightarrow\;
\bar W^{(2)}=\bar z\,h^{(1)\top},\quad
\bar b^{(2)}=\bar z,
$$

$$
\bar h^{(1)}=W^{(2)\top}\bar z
\;\rightarrow\;
\bar a^{(1)}=\bar h^{(1)}\odot\mathrm{SiLU}'(a^{(1)}),
$$

$$
\bar W^{(1)}=\bar a^{(1)}x^{\top},
\qquad
\bar b^{(1)}=\bar a^{(1)},
\qquad
\bar x=W^{(1)\top}\bar a^{(1)} .
$$

### 6.2 单个注意力头的反向

见 §4.10；把 $\bar O$ 从上游传入即可。多头情形把每个头的 $\bar Q_h,\bar K_h,\bar V_h$ 分别投影回 $u$ 并求和，再把 $\bar W^O$ 由拼接矩阵算出（《10-grad-backprop-optimizer-gpu.md》§4.3(c)）。

### 6.3 完整 Transformer 块

编码器与解码器（含交叉注意力、RoPE、SwiGLU、残差）的逐层公式已在《10-grad-backprop-optimizer-gpu.md》§4.3–4.4 完整推导，这里不再重复；本节提供的是其背后的通用机制。

---

## 7. JAX 语义与官方伪代码的对应

官方 AlphaProof 伪代码的更新链路：

```python
self._loss_grad = jax.grad(_loss_fn)              # 反向模式 AD
grads = self._loss_grad(self.params, batch)       # 一次反向得到全部参数梯度
updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
self.params = optax.apply_updates(self.params, updates)
```

对应关系：

- `jax.grad`：对 $(\theta,\text{batch})\mapsto\mathcal{L}$ 做反向模式 AD，返回参数 pytree 的梯度；内部即 §3 的伴随递推，编译到 XLA（《10-grad-backprop-optimizer-gpu.md》§6.6）；
- `jax.vjp` / `jax.jvp`：显式取 VJP / JVP 的接口，前者用于自定义损失或二阶导数，后者用于前向模式；
- `jax.custom_vjp`：为不可微或需要数值技巧的算子（如采样、归一化的稳定实现）自定义反向规则；
- `jax.stop_gradient`：切断梯度（§4.11）；
- `optax`：只负责 §13 的优化器更新，与 AD 解耦。

---

## 8. 梯度检验与数值问题

### 8.1 有限差分检验

中心差分：

$$
\frac{\partial\mathcal{L}}{\partial\theta_i}
\approx
\frac{\mathcal{L}(\theta+\varepsilon e_i)-\mathcal{L}(\theta-\varepsilon e_i)}{2\varepsilon},
$$

误差 $O(\varepsilon^2)$（一阶前向差分为 $O(\varepsilon)$）。检验指标用相对误差

$$
\mathrm{rel}=\frac{|g_{\text{AD}}-g_{\text{FD}}|}{\max(10^{-8},\ |g_{\text{AD}}|+|g_{\text{FD}}|)},
$$

$\mathrm{rel}\lesssim10^{-5}$（fp32）可接受；bf16 下放宽到 $10^{-2}$ 量级。

### 8.2 常见数值陷阱

- log/softmax/归一化必须做稳定化（《12-softmax-logistic-mlp.md》§7）；
- 除以 $\sqrt{v}+\varepsilon$ 时 $\varepsilon$ 过小会在 Adam 早期放大噪声（《13-gradient-descent-and-optimizers.md》§5.5）；
- 混合精度下上溢产生 inf/NaN，需动态 loss scaling（fp16）或直接使用 bf16 的宽指数范围；
- causal mask 用 $-\infty$ 时，softmax 稳定实现应先减最大值再指数化，避免 $0\cdot\infty$。

---

## 9. 小结

- 反向模式 AD（VJP）是“从标量损失反推全部参数梯度”的通用算法；前向模式（JVP）适合输入少、输出多的场景；
- 复杂度上，反向约等于前向的 2–3 倍算力，代价是保存/重算中间激活；
- 一组常用算子的 VJP 规则（矩阵乘、softmax、交叉熵、归一化、注意力、gather/scatter）足以覆盖 Transformer 的全部反向；
- AlphaProof 的 `jax.grad` 与 `optax.adam` 正是“反向模式 AD + 对角预条件一阶优化器”的组合，工程落点见《10-grad-backprop-optimizer-gpu.md》第 6 部分。
