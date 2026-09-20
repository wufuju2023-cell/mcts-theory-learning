# 策略与价值网络的梯度下降：损失函数、逐层反向传播、优化器与 GPU/TPU 内核

参考：T. Hubert et al., *Olympiad-level formal mathematical reasoning with reinforcement learning*, Nature 651, 607–613 (2026), doi:10.1038/s41586-025-09833-y（开放获取）。官方伪代码：Supplementary Data 1 的 `pseudocode.py`；官方补充表：Supplementary Tables 1–7。两份材料已放在本机 `/home/zhai/project/alphaproof-official-materials/`。

本文只补“梯度怎么算、权重矩阵怎么更新、矩阵运算怎么落到底层”这一段。前向结构、搜索与 RL 理论不重复，直接引用已有材料：

- 本目录《6-alpha-proof-mcts-rl.md》§6–7：两项损失、专家迭代、网络总览；
- 本目录《9-mainrl-ttrl-llm-value-head.md》§2：复现路线、两时间尺度、价值误差界；
- 本机笔记 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/10-transformers.md`：前向公式（注意力、RMSNorm、SwiGLU、RoPE、价值头）；
- 同上 `12-rl-math.md`：贝尔曼、蒙特卡洛、策略梯度定理；
- 同上 `04-pretrain-sft.md`、`06-main-rl.md`：预训练、SFT、主 RL 配方；
- 同上 `refs-alpha-proof-paper-nature-2025.md`：论文全文抄录。

本机另已克隆官方示例仓库到 `/home/zhai/project/`：`formal-imo`、`miniF2F`、`alphaproof-nexus-results`。

## 0. 先给结论

- **优化器是 Adam**。官方伪代码 `Network.__init__` 中是 `self.optimizer = optax.adam(config.lr)`，梯度用 `jax.grad`（反向模式自动微分）计算，更新用 `optimizer.update` 加 `optax.apply_updates`；伪代码层面没有权重衰减、没有梯度裁剪。
- **损失**：$\mathcal{L}=\mathcal{L}_{\text{pol}}+\lambda\,\mathcal{L}_{\text{val}}$，Supplementary Table 6 给出策略损失权重 $1.0$、价值损失权重 $\lambda=10^{-3}$、batch size 为 $4096$、回放池 $6\times 10^{7}$ 条、主 RL 约 $10^{6}$ 步。
- **论文与补充材料没有公布**学习率、$\beta_1,\beta_2,\varepsilon$ 与学习率调度；公开伪代码里 `make_config()` 的 `lr=1.0`、`batch_size=2048` 等是演示用的占位值，与 Table 6 的正式值不同。
- **原系统训练在 TPU 上**（论文 Fig. 3 注明 Google Cloud TPU v6e；伪代码 `import jax, optax`），走 JAX/XLA 技术栈，不是 GPU 集群。GPU 上的等价底层算法在第 6 部分给出：cuBLAS/tensor core 的 GEMM、FlashAttention、融合核与 NCCL 集合通信。
- 每步更新的数学骨架是：搜索产生的成功轨迹给出监督信号，反向模式 AD 对全网络（共享编码器 + 策略解码器 + 价值 MLP）求出每一个权重矩阵的梯度，两个头的梯度在共享编码器处相加，Adam 用一阶矩与二阶矩的对角预条件更新参数。

---

## 1. 训练信号与损失函数：精确定义

### 1.1 数据从哪里来

Only-verified 数据。actor 每次尝试得到三类结局：`proved`、`disproved`、`timeout`。只有前两类（且通过最终内核检查 `final_check`）的搜索树会被 `ReplayBuffer.save_game` 接收；失败与超时完全不产生梯度。伪代码的逻辑是：

```python
if game.root.is_optimal:
    replay_buffer.save_game(game)
```

从一棵成功树中抽取的样本是三元组 $(s_t,\ a_t,\ G_t)$：

- $s_t$：pretty-print 并规范化后的 Lean 战术状态；
- $a_t$：成功路径上实际执行的那条战术字符串；
- $G_t$：该状态下搜索计算出的回报（“剩余步数的负值”）。

主 RL 的 batch 按 $90\%$ 自生成 + $10\%$ Mathlib SFT 混合（Table 6：`Fraction SFT = 10%`；TTRL 阶段是 $25\%$）。回放窗口 $6\times 10^{7}$ 条（TTRL 为 $2\times 10^{7}$）。

### 1.2 价值目标：成功树上的蒙特卡洛回报

官方伪代码 `compute_value_target` 的递归定义是：

$$
G(s)=
\begin{cases}
0, & s\ \text{终止（目标全部关闭）},\\[2pt]
-1+G\big(s'(a^\star)\big), & s\ \text{是 OR 节点},\ s'(a^\star)=\text{最优子节点},\\[2pt]
\min_{j\le N} G\big(s^{(j)}\big), & s\ \text{是 AND 节点（拆成 } N \text{ 个子目标）}.
\end{cases}
$$

沿单链 $s_0\to s_1\to\cdots\to s_T$（$T$ 步战术关闭全部目标）它就是

$$
G(s_t)=-(T-t),
$$

即“还剩几步”。AND 节点取 min 等价于“最长分支的负步数”，其激励分析见第 6 篇 §1.2。注意价值目标是**搜索实际找到的回报**（Monte Carlo，不做 bootstrap），这一点决定了价值头是一个条件分布的极大似然问题，而不是 TD 问题。

### 1.3 策略损失（专家迭代）

记策略为自回归解码器

$$
\pi_\theta(a\mid s)=\prod_{j=1}^{|a|} p_\theta\big(a_j\mid a_{<j},\, s\big),
\qquad
p_\theta(\cdot\mid a_{<j},s)=\mathrm{softmax}\big(z_j\big),
$$

其中 $z_j\in\mathbb{R}^{|V|}$ 是第 $j$ 个位置的输出 logits，$s$ 经共享编码器进入解码器的交叉注意力。对一条成功轨迹上的每个状态，策略损失是**序列交叉熵**：

$$
\mathcal{L}_{\text{pol}}^{(i)}
=-\frac{1}{|a^{(i)}|}\sum_{j=1}^{|a^{(i)}|}\log p_\theta\big(a^{(i)}_j\mid a^{(i)}_{<j},s^{(i)}\big).
$$

官方伪代码对应 `optax.softmax_cross_entropy_with_integer_labels(policy_logits, actions)` 再取 `jnp.mean`，即按 token 取平均。它与策略梯度的等价性已在第 6 篇 §6.2 与备份笔记 `12-rl-math.md` §7 证明，结论是

$$
\nabla_\theta\mathcal{L}_{\text{pol}}
=-\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot 1
\;\equiv\;
\text{以“位于成功路径上”为优势的策略梯度},
$$

因为每条进入训练集的轨迹都有确定的正优势（成功），失败被剪掉，所以没有裁剪、没有组相对优势，隐式信任域由“数据总是当前 $\theta$ 搜索出来”的近似 on-policy 性保证。

### 1.4 价值损失（分类交叉熵）

价值头输出 $B=64$ 个 bin 上的分类分布（Supplementary Table 2：`Value: number of bins = 64`）：

$$
p_v(\cdot\mid s)=\mathrm{softmax}\big(\mathrm{MLP}(\bar h(s))\big)\in\Delta^{B-1},
\qquad
\widehat V(s)=-\sum_{b=0}^{B-1} m_b\,p_v(b\mid s),
$$

$m_b$ 是第 $b$ 个 bin 的中点，bin 覆盖剩余步数 $d\in[0,D_{\max}]$（论文只说明是分类分布；对数间隔分箱与 $D_{\max}$ 属重建细节）。对三元组 $(s,a,G)$，令 $y=\mathrm{bin}(G)$（或等价地取 $y=\mathrm{bin}(d)$，$d=-G$），则

$$
\mathcal{L}_{\text{val}}^{(i)}=-\log p_v\big(y^{(i)}\mid s^{(i)}\big).
$$

其统计含义是极大似然拟合经验回报分布，最小值点满足 $\widehat V(s)\to\mathbb{E}[G\mid s]=-d(s)$，即价值误差通过 $Q$ 变换的乘性阻尼进入搜索（定量界见第 9 篇 §2.3.2、§2.3.3）。

### 1.5 总损失

$$
\boxed{\;
\mathcal{L}(\theta)
=\frac{1}{n}\sum_{i=1}^{n}
\Big[
\mathcal{L}_{\text{pol}}^{(i)}
+\lambda\,\mathcal{L}_{\text{val}}^{(i)}
\Big],
\qquad \lambda=10^{-3}.
\;}
$$

$\theta$ 包含三部分：共享编码器参数 $\theta_{\text{enc}}$、策略解码器参数 $\theta_{\text{dec}}$（含共享的词嵌入与解嵌矩阵）、价值 MLP 参数 $\psi$。因为两个头共享编码器，编码器收到的是两路梯度的和：

$$
\nabla_{\theta_{\text{enc}}}\mathcal{L}
=\nabla_{\theta_{\text{enc}}}\mathcal{L}_{\text{pol}}
+\lambda\,\nabla_{\theta_{\text{enc}}}\mathcal{L}_{\text{val}}.
$$

$\lambda=10^{-3}$ 的作用是把价值损失的梯度量级压到与策略损失可比，避免价值头（一个 64 类小分类器）夺走共享表示。

---

## 2. 反向模式自动微分：把标量损失变成所有参数的梯度

### 2.1 伴随（adjoint）记号

把前向计算写成有向无环图：节点 $u$ 是张量，边是算子。对图里任意中间张量 $v$，定义伴随

$$
\bar v \;:=\; \frac{\partial \mathcal{L}}{\partial v},
$$

形状与 $v$ 相同。反向传播就是从 $\bar{\mathcal{L}}=1$ 出发，沿拓扑逆序把伴随从输出推回输入。

### 2.2 三条基本规则

**(R1) 线性算子。** 设 $Y=XW$（$X\in\mathbb{R}^{n\times d}$，$W\in\mathbb{R}^{d\times k}$）。由 $\mathrm{d}Y=\mathrm{d}X\,W+X\,\mathrm{d}W$ 与 $\mathrm{d}\mathcal{L}=\langle \bar Y,\mathrm{d}Y\rangle$：

$$
\bar X=\bar Y\,W^{\top},
\qquad
\bar W=X^{\top}\bar Y .
$$

这是整个反向传播里最贵的步骤：两个矩阵乘，形状分别是 $(n\times k)(k\times d)$ 与 $(d\times n)(n\times k)$。

**(R2) 逐元素算子。** 设 $Y=f(X)$ 逐元素作用，则

$$
\bar X=\bar Y\odot f'(X),
$$

其中 $\odot$ 是 Hadamard 积，$f'$ 逐元素求值。

**(R3) 分支与残差。** 若 $u$ 同时喂给多个算子，则

$$
\bar u=\sum_{\text{branches}}\bar u_{\text{branch}},
$$

残差连接 $x'=x+F(x)$ 因此是“直通梯度 + 支路梯度”。

反向模式的开销是前向的常数倍（每个线性层两次 GEMM，每个非线性一次廉价逐元素运算）；对“一个标量损失、$3\times10^{9}$ 个参数”的局面，这是唯一可行的选择。JAX 里这些由 `jax.grad` 自动完成：它把 Python 函数变换成返回参数 pytree 梯度的函数，官方伪代码只用一行

```python
self._loss_grad = jax.grad(_loss_fn)
```

之后每次 `network.update(batch)` 都调用 `_loss_grad(self.params, batch)`。JAX 的追踪把上述规则编译成 XLA 计算图（第 6.6 节）。

---

## 3. 前向图的紧凑形式（反传的对照对象）

完整推导见备份 `10-transformers.md`。这里只列反传要用到的公式，统一记号：

- $d=2048$（$=16\times128$，Supplementary Table 2：16 头、每头 128 维）；
- $N_{\text{enc}}=18$ 个编码器块，$N_{\text{dec}}=24$ 个解码器块（同上）；
- FFN 加宽系数 $6$；$B=64$ 个价值 bin；每次扩展采样 $K=6$ 条战术（同上）；
- $L_s$ 为状态 token 数（预训练编码器侧 1024），$L_a$ 为战术 token 数（训练时 $\le 64$）。

### 3.1 编码器（单向，无掩码）

输入 $h^{(0)}=\mathrm{Emb}[x]+\mathrm{Pos}$，第 $\ell$ 块：

$$
u=\mathrm{RMSNorm}\big(h^{(\ell)}\big),
$$

$$
Q_h=uW_h^{Q},\quad K_h=uW_h^{K},\quad V_h=uW_h^{V},
\qquad
S_h=\frac{\widehat{Q}_h\widehat{K}_h^{\top}}{\sqrt{d_k}},
\qquad
A_h=\mathrm{softmax}_{\text{row}}(S_h),
$$

其中 $\widehat{Q}_h,\widehat{K}_h$ 是施加 RoPE 之后的版本（位置信息只在 $Q,K$ 上旋转；RoPE 细节见备份 `10-transformers.md` §6）。然后

$$
\mathrm{MHA}(u)=\mathrm{Concat}_h(A_hV_h)\,W^{O},
\qquad
a=h^{(\ell)}+\mathrm{MHA}(u),
$$

$$
z=\mathrm{RMSNorm}(a),
\qquad
\mathrm{FFN}(z)=\Big(\big(zW_1\big)\odot\mathrm{SiLU}\big(zW_g\big)\Big)W_2,
$$

$$
h^{(\ell+1)}=a+\mathrm{FFN}(z).
$$

编码器输出 $\hat H=h^{(N_{\text{enc}})}\in\mathbb{R}^{L_s\times d}$，摘要向量 $\bar h=\hat H[0]$。

### 3.2 价值头

$$
g=\mathrm{RMSNorm}(\bar h),
\qquad
a_v=\mathrm{SiLU}(W_{v1}g+b_{v1}),
\qquad
z_v=W_{v2}a_v+b_{v2}\in\mathbb{R}^{B},
\qquad
p_v=\mathrm{softmax}(z_v).
$$

（隐藏宽度论文未公布，这里写成一般形式；一层的 SiLU MLP 属于重建。）

### 3.3 解码器（策略）

第 $j$ 步输入 token $a_{<j}$ 经共享嵌入得到 $e_j$，解码器块依次做：因果自注意力、对 $\hat H$ 的交叉注意力、SwiGLU FFN，全部 pre-norm。最后经输出投影（与输入嵌入绑定）得到 logits：

$$
z_j=W_E\,d_j^{(N_{\text{dec}})},
\qquad
p_\theta(a_j\mid a_{<j},s)=\mathrm{softmax}(z_j)_{a_j}.
$$

---

## 4. 逐层反向传播：完整公式推导

反向传播的顺序与第 3 节完全相反：先价值头与策略头各自的最后一层，再逐块回穿解码器/价值 MLP，最后汇入编码器。下面所有伴随记号 $\delta_X=\partial\mathcal{L}/\partial X$ 都指对总损失 $\mathcal{L}$ 求导（价值路径上含因子 $\lambda$）。为避免符号爆炸，先给 4 个可复用的引理，再按网络顺序组装。

### 4.1 四个引理

#### 引理 1（softmax 的 Jacobian）

对一行 $x\in\mathbb{R}^{n}$，$p=\mathrm{softmax}(x)$。则

$$
\frac{\partial p_i}{\partial x_j}=p_i\big(\delta_{ij}-p_j\big).
$$

**推导。** $p_i=e^{x_i}/\sum_k e^{x_k}$。若 $i=j$：

$$
\frac{\partial p_i}{\partial x_i}
=\frac{e^{x_i}\sum_k e^{x_k}-e^{x_i}e^{x_i}}{\big(\sum_k e^{x_k}\big)^2}
=p_i-p_i^2=p_i(1-p_i);
$$

若 $i\ne j$：

$$
\frac{\partial p_i}{\partial x_j}
=-\frac{e^{x_i}e^{x_j}}{\big(\sum_k e^{x_k}\big)^2}=-p_i p_j.
$$

合并即得。把 Jacobian 作用到行伴随 $\bar p$ 上：

$$
\bar x_i
=\sum_j \bar p_j\frac{\partial p_j}{\partial x_i}
=\sum_j \bar p_j\,p_j(\delta_{ij}-p_i)
=p_i\Big(\bar p_i-\sum_j p_j\bar p_j\Big).
$$

写成矩阵形式（按行独立）：

$$
\boxed{\;
\bar S=A\odot\Big(\bar A-\big(\mathrm{rowsum}(\bar A\odot A)\big)\mathbf{1}^{\top}\Big),
\;}
$$

其中 $A=\mathrm{softmax}_{\text{row}}(S)$，$\mathrm{rowsum}(M)$ 把每行求和得到列向量，$\mathbf{1}$ 是全 1 列向量。注意：注意力里的 softmax 是**行方向**的，所以上式按行施加，行与行之间不耦合。

#### 引理 2（softmax 加交叉熵）

若 $\mathcal{L}=-\log p_y$（one-hot 目标 $e_y$），则

$$
\frac{\partial\mathcal{L}}{\partial z}
=p-e_y,
\qquad\text{即}\qquad
\bar z_a=p_a-\mathbf{1}[a=y].
$$

**推导。** $\mathcal{L}=-z_y+\log\sum_k e^{z_k}$，直接求导：

$$
\frac{\partial\mathcal{L}}{\partial z_a}
=-\delta_{ay}+\frac{e^{z_a}}{\sum_k e^{z_k}}
=p_a-\delta_{ay}.
$$

这条引理是策略头与价值头共用的“最后一层”梯度，也是交叉熵训练的核心：**把正确类的概率往上推、其余类按当前概率按比例往下压**。

#### 引理 3（RMSNorm）

设

$$
y=\gamma\odot\frac{x}{r},
\qquad
r=\sqrt{\frac{1}{d}\sum_j x_j^2+\varepsilon}.
$$

则对行向量 $x$（每个 token 一行，$\gamma$ 全位置共享）：

$$
\boxed{\;
\delta_x=\frac{1}{r}\Big(\gamma\odot\delta_y\Big)
-\frac{x}{r^{3}}\cdot\frac{1}{d}\Big\langle x,\ \gamma\odot\delta_y\Big\rangle,
\;}
\qquad
\delta_\gamma=\sum_{\text{positions}}\delta_y\odot\frac{x}{r}.
$$

**推导。** $y_i=\gamma_i x_i/r$，$r=r(x)$。求偏导：

$$
\frac{\partial y_i}{\partial x_j}
=\frac{\gamma_i}{r}\delta_{ij}-\frac{\gamma_i x_i}{r^2}\frac{\partial r}{\partial x_j},
\qquad
\frac{\partial r}{\partial x_j}=\frac{x_j}{d\,r}.
$$

于是

$$
\delta_{x_j}
=\sum_i \delta_{y_i}\frac{\partial y_i}{\partial x_j}
=\frac{\gamma_j\delta_{y_j}}{r}
-\frac{x_j}{d\,r^{3}}\sum_i \gamma_i x_i\delta_{y_i}
=\frac{(\gamma\odot\delta_y)_j}{r}
-\frac{x_j}{d\,r^{3}}\big\langle x,\gamma\odot\delta_y\big\rangle .
$$

第二项是“全局收缩项”：所有坐标的梯度都会按 $x$ 的方向减去一个分量，这正是归一化算子耦合各坐标的体现。$\delta_\gamma$ 由 $y_i=\gamma_i(\cdot)$ 直接得，并在序列维求和使用同一 $\gamma$ 的位置。

#### 引理 4（注意力）

对单个头，$S=QK^{\top}/\sqrt{d_k}$，$A=\mathrm{softmax}_{\text{row}}(S)$，$O=AV$。给定 $\delta_O$：

$$
\boxed{\;
\delta_V=A^{\top}\delta_O,
\qquad
\delta_A=\delta_O V^{\top},
\qquad
\delta_S=A\odot\big(\delta_A-\mathrm{rowsum}(\delta_A\odot A)\mathbf{1}^{\top}\big),
\;}
$$

$$
\boxed{\;
\delta_Q=\frac{1}{\sqrt{d_k}}\,\delta_S K,
\qquad
\delta_K=\frac{1}{\sqrt{d_k}}\,\delta_S^{\top} Q.
\;}
$$

**推导。** $O_{ij}=\sum_k A_{ik}V_{kj}$，故

$$
\delta_{V_{kj}}=\sum_i \delta_{O_{ij}}A_{ik}\ \Rightarrow\ \delta_V=A^{\top}\delta_O,
\qquad
\delta_{A_{ik}}=\sum_j\delta_{O_{ij}}V_{kj}\ \Rightarrow\ \delta_A=\delta_O V^{\top}.
$$

$\delta_S$ 由引理 1。$\delta_Q,\delta_K$ 来自 $S_{ij}=(\sum_a Q_{ia}K_{ja})/\sqrt{d_k}$：

$$
\delta_{Q_{ia}}=\sum_j\delta_{S_{ij}}\frac{K_{ja}}{\sqrt{d_k}},
\qquad
\delta_{K_{ja}}=\sum_i\delta_{S_{ij}}\frac{Q_{ia}}{\sqrt{d_k}},
$$

即上述矩阵形式。加上 RoPE 之后，$\delta$ 还要经过旋转矩阵的转置：若 $\widehat q=R_m q$，则 $\delta_q=R_m^{\top}\delta_{\widehat q}$（每个二维对子做反向旋转，角度相反）。

#### 引理 5（SiLU 与 SwiGLU）

$$
f(x)=x\,\sigma(x),
\qquad
f'(x)=\sigma(x)+x\,\sigma(x)\big(1-\sigma(x)\big)
=\sigma(x)\big(1+x(1-\sigma(x))\big),
$$

其中 $\sigma$ 是 sigmoid。对 $\mathrm{FFN}(z)=(P\odot\mathrm{SiLU}(G))W_2$（记 $P=zW_1$，$G=zW_g$，$C=P\odot\mathrm{SiLU}(G)$）：

$$
\delta_C=\delta_Y W_2^{\top},
\qquad
\delta_P=\delta_C\odot\mathrm{SiLU}(G),
\qquad
\delta_G=\delta_C\odot P\odot \mathrm{SiLU}'(G),
$$

$$
\delta_{W_2}=C^{\top}\delta_Y,
\qquad
\delta_{W_1}=z^{\top}\delta_P,
\qquad
\delta_{W_g}=z^{\top}\delta_G,
\qquad
\delta_z=\delta_P W_1^{\top}+\delta_G W_g^{\top}.
$$

### 4.2 价值头反传（先算，因为它在编码器出口处）

设 batch 中一条样本的目标 bin 为 $y$。

**(a) logits 梯度。** 由引理 2：

$$
\delta_{z_v}=p_v-e_y\in\mathbb{R}^{B}.
$$

**(b) MLP 的反传。** 记 $u=W_{v1}g+b_{v1}$，$a_v=\mathrm{SiLU}(u)$，$z_v=W_{v2}a_v+b_{v2}$：

$$
\delta_{W_{v2}}=\delta_{z_v}a_v^{\top},
\qquad
\delta_{b_{v2}}=\delta_{z_v},
\qquad
\delta_{a_v}=W_{v2}^{\top}\delta_{z_v},
$$

$$
\delta_u=\delta_{a_v}\odot\mathrm{SiLU}'(u),
\qquad
\delta_{W_{v1}}=\delta_u g^{\top},
\qquad
\delta_{b_{v1}}=\delta_u,
\qquad
\delta_g=W_{v1}^{\top}\delta_u .
$$

**(c) 回穿 RMSNorm。** 把引理 3 用在 $g=\mathrm{RMSNorm}(\bar h)$ 上：

$$
\delta_{\bar h}
=\frac{1}{r}\big(\gamma_v\odot\delta_g\big)
-\frac{\bar h}{r^{3}}\cdot\frac{1}{d}\big\langle \bar h,\ \gamma_v\odot\delta_g\big\rangle,
\qquad
\delta_{\gamma_v}=\delta_g\odot\frac{\bar h}{r}.
$$

$\delta_{\bar h}$ 只落在编码器输出的第 0 个位置上，成为编码器反向的“价值入口”。

**(d) 两个评估用的小结论**（不进反传，但说明价值头的几何）。

$\widehat V=-\langle m,p_v\rangle$ 对 logits 的导数：

$$
\frac{\partial \widehat V}{\partial z_{v,c}}
=-\sum_b m_b\,p_{v,b}(\delta_{bc}-p_{v,c})
=p_{v,c}\Big(\sum_b m_b p_{v,b}-m_c\Big)
=-p_{v,c}\big(m_c-\bar m\big),
\qquad
\bar m:=\sum_b m_b p_{v,b}=-\widehat V .
$$

即

$$
\frac{\partial\widehat V}{\partial z_v}=-\,p_v\odot\big(m+\widehat V\,\mathbf{1}\big).
$$

它把“分类分布的更新”和“标量价值的变化”联系起来：价值向 bin $m_c$ 移动的速率是 $p_{v,c}(m_c+\widehat V)$，与第 9 篇讨论的 $\gamma^{k}$ 阻尼配套。

### 4.3 策略头反传

#### (a) 序列交叉熵的 logits 梯度

对每个解码位置 $j$（只对战术 token 位置，padding 掩掉）：

$$
\delta_{z_j}=p_\theta(\cdot\mid a_{<j},s)-e_{a_j}\in\mathbb{R}^{|V|}.
$$

若损失是 $|a|$ 个位置的平均，则整体乘 $1/|a|$。这与策略梯度法的形式对照是：

$$
\nabla_\theta\mathcal{L}_{\text{pol}}
=-\frac{1}{|a|}\sum_j\nabla_\theta\log p_\theta(a_j\mid\cdot)
=-\frac{1}{|a|}\sum_j\Big(\frac{\partial z_j}{\partial\theta}\Big)^{\top}\delta_{z_j},
$$

也就是说，反向 AD 自动实现了“优势恒为 1”的 REINFORCE（第 1.3 节）。

#### (b) 解嵌与嵌入（权重绑定）

设输出投影 $z_j=W_E d_j$，$W_E\in\mathbb{R}^{|V|\times d}$。由引理 1 的线性版本：

$$
\delta_{d_j}=W_E^{\top}\delta_{z_j},
\qquad
\delta_{W_E}\mathrel{+}=\sum_j \delta_{z_j}\,d_j^{\top}.
$$

输入侧（战术 token 的嵌入查表）反向是“散射-累加”：对 token $v$，

$$
\delta_{W_E}[v]\mathrel{+}=\sum_{j:\ a_{<j}=v}\delta_{e_j}.
$$

权重绑定意味着 $\delta_{W_E}$ 同时收到输出投影与输入查表两路散射，必须相加（R3）。

#### (c) 解码器块反传（逐块从 $N_{\text{dec}}-1$ 到 0）

每个 pre-norm 块内部的反向顺序是前向的镜像：

1. **FFN 支路**：$a\to z=\mathrm{RMSNorm}(a)\to\mathrm{FFN}(z)$，用引理 3、5 得 $\delta_z^{\text{FFN}}$；
2. **残差汇合**：$\delta_a=\delta_{a+\text{FFN}}+\delta_z^{\text{FFN}}$ 的 RMSNorm 回传；
3. **交叉注意力支路**：$a\to u=\mathrm{RMSNorm}(a)\to$ 交叉注意。$Q$ 来自解码器，$K,V$ 来自编码器输出 $\hat H$：

$$
\delta_{Q_h}=\big(\delta_S\widehat K_h\big)/\sqrt{d_k},
\qquad
\delta_{K_h}=\delta_S^{\top}\widehat Q_h/\sqrt{d_k},
\qquad
\delta_{V_h}=A_h^{\top}\delta_O,
$$

$$
\delta_{W_h^{Q}}=\big(\mathrm{RoPE}^{\top}\delta_{Q_h}\big)^{\top}u,
\qquad
\delta_{W_h^{K}}=\big(\mathrm{RoPE}^{\top}\delta_{K_h}\big)^{\top}u,
\qquad
\delta_{W_h^{V}}=\delta_{V_h}^{\top}u.
$$

编码器侧的伴随是每个头、每个解码层的累加：

$$
\delta_{\hat H}\mathrel{+}=\mathrm{RoPE}^{\top}\!\big(\delta_{K_h}\big)W_h^{K\ \top}
+\delta_{V_h}W_h^{V\ \top}.
$$

4. **自注意力支路**：同引理 4，但 $S$ 需加因果掩码（掩掉的位置概率为 0，反传自然为 0，因为 $A$ 在那里是 0）；
5. **残差汇合**：得到该块输入的 $\delta$，继续下一块。

多头拼接与输出投影：

$$
\delta_{O_{\text{cat}}}=\delta_Y W^{O\ \top},
\qquad
\delta_{W^{O}}=O_{\text{cat}}^{\top}\delta_Y,
\qquad
\delta_{O_h}=\big(\delta_{O_{\text{cat}}}\big)_{\text{slice }h}.
$$

### 4.4 编码器反传

编码器收到的伴随有两路（这就是“共享编码器”的数学体现）：

$$
\delta_{\hat H}
=\underbrace{\sum_{\ell=0}^{N_{\text{dec}}-1}\sum_{h}\Big(\mathrm{RoPE}^{\top}\delta_{K_h}^{(\ell)}W_h^{K\ \top}+\delta_{V_h}^{(\ell)}W_h^{V\ \top}\Big)}_{\text{策略解码器交叉注意力}},
\;+\;
\underbrace{e_0\,\delta_{\bar h}^{\top}}_{\text{价值头，只作用于第 0 个位置}},
$$

其中 $e_0$ 是“第 0 位置”的选择向量。之后按编码器块的镜像顺序反传（与 4.3(c) 相同，只是没有因果掩码、也没有交叉注意力），最终得到

$$
\delta_{W_E}[v]\mathrel{+}=\sum_{i:\ x_i=v}\delta_{h^{(0)}_i},
$$

以及每层的 $\delta_{W^{Q}_h},\delta_{W^{K}_h},\delta_{W^{V}_h},\delta_{W^{O}},\delta_{W_1},\delta_{W_g},\delta_{W_2},\delta_{\gamma}$ 与所有偏置。

### 4.5 汇总：一次反向传播的伪代码

```python
# 前向（对 batch 中每条 (s, a, G)）
H = encoder(x(s))                      # 18 个 pre-norm 块
hbar = H[0]
zv = value_mlp(rmsnorm(hbar))          # 64 维 logits
zv_seq, dec_cache = decoder(a, H)      # 24 个块，因果自注意力 + 交叉注意力
L = cross_entropy(zv, bin(G)) * 1e-3 + cross_entropy_seq(zv_seq, a) / len(a)

# 反向（jax.grad 自动做，这里写出显式顺序）
dzv = softmax(zv) - onehot(bin(G))              # 引理 2
dpsi, dg = backprop_value_mlp(dzv)              # 4.2
dH = zeros_like(H); dH[0] += backprop_rmsnorm(dg)

dz_seq = softmax(zv_seq) - onehot(a)            # 4.3(a)
dW_E = dz_seq^T @ d_emb                         # 输出投影散射
dd = dW_E @ dz_seq                              # 解码器输入伴随
for layer in reversed(decoder_layers):          # 4.3(c)
    dd, dH = backprop_decoder_block(layer, dd, dH)
for layer in reversed(encoder_layers):          # 4.4
    dH, dtheta_enc = backprop_encoder_block(layer, dH)
dW_E += scatter_add(dH_emb)                     # 权重绑定
```

### 4.6 梯度量的形状与成本

对单个线性层 $Y=XW$（$X$ 是 $T\times d$，$W$ 是 $d\times k$）：

- 前向：$2Tdk$ FLOPs；
- $\delta_X=\delta_YW^{\top}$：$2Tdk$ FLOPs；
- $\delta_W=X^{\top}\delta_Y$：$2Tdk$ FLOPs。

反向是前向的两倍算力，这正是“每 token 前向 $2N$、训练总耗 $6N$”的来源（$N$ 为参数量）。注意力矩阵部分与序列长度平方相关，单独计数（第 6.1 节）。

---

## 5. 参数更新：优化器、裁剪与调度

### 5.1 官方事实

官方伪代码的整个更新循环是：

```python
self.optimizer = optax.adam(config.lr)          # Config.__init__ 前已设 self.lr
...
self._loss_grad = jax.grad(_loss_fn)
...
grads = self._loss_grad(self.params, batch)
updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
self.params = optax.apply_updates(self.params, updates)
```

对应的训练事实：

- **优化器是 Optax 的 Adam**（`optax.adam`），不是 `optax.adamw`，伪代码层面没有解耦权重衰减；也没有梯度裁剪、没有 EMA 目标网络。
- 伪代码 `Config` 给出 `value_weight = 0.001`，与 Supplementary Table 6 的 `Value Loss Weight = 1e-3`、`Policy Loss Weight = 1.0` 一致。
- `make_config()` 里的 `lr=1.0`、`batch_size=2048`、`num_simulations=800`、`num_actors=3000` 是演示配置；正式 batch 是 4096（Table 6）。
- 学习率、$\beta_1,\beta_2,\varepsilon$、调度、是否裁剪**均未在论文与补充表中公布**；`optax.adam` 的默认值（下文）是唯一有官方代码依据的部分。

### 5.2 Adam 的精确公式与偏差修正

令第 $k$ 步的随机梯度为 $g_k=\nabla_\theta\mathcal{L}(\theta_{k-1})$（在 batch 上求平均，多设备时先做 all-reduce 平均，见 6.5）。Adam 维护一阶矩与二阶矩：

$$
m_k=\beta_1 m_{k-1}+(1-\beta_1)\,g_k,
\qquad
v_k=\beta_2 v_{k-1}+(1-\beta_2)\,g_k\odot g_k,
\qquad
m_0=v_0=0 .
$$

**偏差修正的推导。** 展开一阶矩：

$$
m_k=(1-\beta_1)\sum_{i=1}^{k}\beta_1^{\,k-i}g_i .
$$

若梯度的均值在窗口内近似平稳（$\mathbb{E}[g_i]=g$），则

$$
\mathbb{E}[m_k]=(1-\beta_1)\Big(\sum_{i=1}^{k}\beta_1^{\,k-i}\Big)g
=(1-\beta_1)\frac{1-\beta_1^{k}}{1-\beta_1}g
=\big(1-\beta_1^{k}\big)g .
$$

所以 $m_k$ 偏小 $1-\beta_1^{k}$ 倍；二阶矩同理偏小 $1-\beta_2^{k}$ 倍。定义

$$
\widehat m_k=\frac{m_k}{1-\beta_1^{k}},
\qquad
\widehat v_k=\frac{v_k}{1-\beta_2^{k}},
$$

就得到无偏的矩估计。参数更新为

$$
\boxed{\;
\theta_k=\theta_{k-1}-\eta_k\,
\frac{\widehat m_k}{\sqrt{\widehat v_k}+\varepsilon},
\;}
$$

（根号与除法逐坐标进行。）Optax 的 `scale_by_adam` 默认 $\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=10^{-8}$，$v$ 的偏差修正按上述实现；`scale_by_learning_rate` 默认带负号，因此 `apply_updates` 只是加法。

### 5.3 数学解读：Adam 做了什么

把更新改写成

$$
\Delta\theta_k=-\eta_k\,D_k\,\widehat m_k,
\qquad
D_k=\mathrm{diag}\Big(\big(\sqrt{\widehat v_k}+\varepsilon\big)^{-1}\Big),
$$

可见 Adam 是**对角预条件的梯度下降**：

- 每个坐标的有效步长与该坐标历史梯度的 RMS 成反比，参数尺度不同的坐标（嵌入、LayerNorm 的 $\gamma$、大的注意力矩阵）被自动拉到同一尺度，这对 3B 参数、异构层的模型的稳定性很关键；
- 当某个坐标的梯度符号长期一致时 $\widehat m/\sqrt{\widehat v}\approx \pm1$，每步约走 $\eta_k$，表现为“近似 sign 方法”；
- 与二阶方法的联系：$D_k$ 是海森对角（或 Fisher 对角）的粗略估计；真正的二阶方法（K-FAC、Shampoo）会用矩阵型预条件子，代价高得多，在 $3\times10^9$ 参数规模上未被采用；
- Adam 的常见问题与对策（重建层的建议）：需要权重衰减时用解耦形式

$$
\theta_k=\theta_{k-1}-\eta_k\Big(\frac{\widehat m_k}{\sqrt{\widehat v_k}+\varepsilon}+\lambda_{\text{wd}}\,\theta_{k-1}\Big),
$$

即 AdamW；通常对 LayerNorm 参数、偏置与嵌入不施加衰减。官方伪代码没有这一项，若要照论文复现主 RL，应保持“无 weight decay 的 Adam + 其它正则（10% SFT 混合、dropout 等）”。

### 5.4 梯度裁剪与学习率调度（未公布，给可复现的重建）

虽然伪代码没有裁剪，$10^{6}$ 步的 RL 训练通常需要全局范数裁剪来吸收搜索分布漂移造成的偶发大梯度。标准公式是

$$
g\leftarrow g\cdot\min\Big(1,\ \frac{c}{\|g\|_2}\Big),
\qquad
\|g\|_2=\sqrt{\sum_{p}\|g_p\|_F^2},
$$

$c$ 常取 $1.0$（对 $\theta$ 的全部矩阵逐元素平方和开根号）。裁剪后再送入 Adam。

学习率调度同样未公布。可复现的重建形式是线性 warmup 加余弦衰减：

$$
\eta_k=
\begin{cases}
\eta_{\max}\dfrac{k}{k_{\text{warm}}}, & k\le k_{\text{warm}},\\[8pt]
\eta_{\min}+\dfrac{\eta_{\max}-\eta_{\min}}{2}
\Big(1+\cos\Big(\pi\dfrac{k-k_{\text{warm}}}{K-k_{\text{warm}}}\Big)\Big), & k>k_{\text{warm}},
\end{cases}
$$

其中 $K\approx10^{6}$ 是主 RL 总步数；$\eta_{\max}\sim10^{-4}$ 到 $3\times10^{-4}$、warmup $5\times10^{3}$ 步、$\eta_{\min}\sim\eta_{\max}/10$ 是 3B 模型的常见量级（备份 `06-main-rl.md` §4 曾给过 AdamW、$3\times10^{-5}\to3\times10^{-6}$ 的旧猜测；以官方伪代码为准，优化器应为 Adam）。是否使用调度、以及 $\eta_{\max}$ 的精确值是当前最大的未知量之一，建议在复现实验 E1/E4 中使用相同的总步数与 batch 做一次小规模扫描。

### 5.5 输出层的数值细节

- 交叉熵实现用“logits + 整数标签”（`softmax_cross_entropy_with_integer_labels`），内部做 log-sum-exp 稳定化：$\log p_y=z_y-\mathrm{LSE}(z)$，避免显式计算 $p$ 再取对数；
- 价值头只有 64 类，它的一次 GEMM 相对整网可忽略；策略头 logits 是 $|V|$ 类的巨大矩阵，往往与损失融合成一个核（6.4 节）；
- 训练用 bf16 存激活、fp32 累加与 fp32 主权重（Adam 的 $m,v$ 也保持 fp32），细节见 6.6 与 5.6。

### 5.6 多设备下的平均

设全局 batch 被切到 $D$ 个设备。由于损失是样本平均、求导与平均可交换，

$$
g=\nabla_\theta\Big(\frac{1}{n}\sum_{i=1}^{n}\ell_i\Big)
=\frac{1}{D}\sum_{d=1}^{D}\nabla_\theta\Big(\frac{1}{n_d}\sum_{i\in\mathcal{I}_d}\ell_i\Big)
=\frac{1}{D}\sum_d g_d .
$$

所以每个设备算本地平均梯度，再 all-reduce 求平均即可，等价于在全局 batch 上做一次 Adam 步。这就是数据并行的数学基础（工程实现见 6.5）。

---

## 6. 底层矩阵算法：从公式到 GPU/TPU 内核

### 6.1 先把网络数成 GEMM

除逐元素运算与 softmax 外，Transformer 的每个权重矩阵都出现在一个 GEMM 里。以 $T$ 个 token（整个 batch 的所有序列拼接）计：

- QKV 投影：$(T\times d)\times(d\times 3d)$（或三个 $d\times d$）；
- 注意力分数：每个头 $(T\times d_k)\times(d_k\times T)$，输出 $T\times T$（与 $T^2$ 成正比，长状态时是瓶颈）；
- 注意力加权：$(T\times T)\times(T\times d_v)$；
- 输出投影：$(T\times H d_v)\times(H d_v\times d)$；
- FFN：$(T\times d)\times(d\times d_{\text{ff}})$ 两次（$W_1$ 与 $W_g$），再 $(T\times d_{\text{ff}})\times(d_{\text{ff}}\times d)$；
- 解码器 logits：$(T_a\times d)\times(d\times|V|)$，词表大时这项很贵；
- 价值头：$(1\times d)\times(d\times m)$ 加 $(1\times m)\times(m\times 64)$，可忽略。

反向对每个 GEMM 再做两次（$\delta_X$ 与 $\delta_W$），因此一次参数更新的算力约为前向的 3 倍。论文的机器算力公式也写成

$$
C_{\text{step}}\approx 6\,N_\theta\cdot(\text{batch tokens}),
$$

$6$ 就是“前向 $2$ + 反向 $4$”的系数（$N_\theta\approx3\times10^{9}$）。

### 6.2 GPU 上的 GEMM：cuBLAS、tensor core 与分块

GPU 实现全网的路径是：

1. **cuBLAS/cuBLASLt（或 CUTLASS）** 提供 GEMM 原语。列主序存储，用 `C = alpha*op(A)*op(B)+beta*C` 的 BLAS 约定；注意 PyTorch 是行主序，实际调用时会转成 TN/NT 变体；
2. **张量核心**：每个 SM 的 MMA 指令（Volta 起 m8n8k4，Ampere m16n8k16 bf16、m16n8k8 tf32，Hopper 的 wgmma 异步指令、Blackwell 的 tcgen05）一次算一个小块矩阵乘。bf16 输入、fp32 累加是混合精度训练的标准配置；fp32 权重路径可以用 TF32（19 位尾数）加速但精度略降；
3. **分块（tiling）**：把大矩阵切成 CTA tile（常见 $128\times128$ 或 $256\times128$），K 维分批装入共享内存，寄存器里做累加。A/B 的复用决定了算术强度：$d=2048$ 的 GEMM 每读一个数做上千次乘加，因此是 compute-bound，能跑满 tensor core；
4. **split-K**：当某个维度很小时（典型是 $\delta_W=X^{\top}\delta_Y$，$T$ 很大而 $k$ 小，或价值头），沿 K 维切分并行、用原子加或二次规约合并，提高 SM 占用；
5. **epilogue 融合**：偏置加法、激活、甚至残差加法都可以融进 GEMM 的收尾阶段，省一次读写（CUTLASS 的 epilogue 机制、cuBLASLt 的 epilogue 选项）；多层融合则由 `torch.compile`/XLA 的算子融合完成（6.6）。

对注意力还可以用**批量 GEMM**（`cublasGemmStridedBatched`）把 $H$ 个头与 batch 维并行起来；这就是“MHA 是一组批量小矩阵乘”的标准做法。

### 6.3 FlashAttention：省内存的注意力与它的反向

朴素实现会显式写出 $T\times T$ 的权重矩阵，显存 $O(T^2)$。FlashAttention 不写 $P$，而是把 $Q,K,V$ 分块驻留 SRAM，用**在线 softmax** 递推：维护行最大值 $m$、行归一化和 $\ell$、输出累加 $o$。备份 `10-transformers.md` §8 已给出前向的合并公式：

$$
m_{\text{new}}=\max(m,m_2),
\qquad
\ell_{\text{new}}=\ell\,e^{\,m-m_{\text{new}}}+l_2\,e^{\,m_2-m_{\text{new}}},
$$

$$
o_{\text{new}}=\frac{o\,e^{\,m-m_{\text{new}}}+o_2\,e^{\,m_2-m_{\text{new}}}}{\ell_{\text{new}}}.
$$

反向传播时，为省显存**不保存 $P$**，而是在反向里重新计算每个块的 $S=QK^{\top}/\sqrt{d_k}$ 与 $P=\exp(S-L)$（$L$ 是前向存下的 log-sum-exp）。关键恒等式（单行 $i$）：

$$
O_i=\sum_j P_{ij}V_j,
\qquad
\Delta_i:=\sum_j P_{ij}\,\mathrm{d}P_{ij}
=\sum_j P_{ij}\,\big(\mathrm{d}O_i\cdot V_j\big)
=\mathrm{d}O_i\cdot O_i,
$$

于是由引理 1，

$$
\mathrm{d}S=P\odot\big(\mathrm{d}O\,V^{\top}-\Delta\,\mathbf{1}^{\top}\big),
\qquad
\mathrm{d}V\mathrel{+}=P^{\top}\mathrm{d}O,
\qquad
\mathrm{d}Q\mathrel{+}=\frac{1}{\sqrt{d_k}}\mathrm{d}S\,K,
\qquad
\mathrm{d}K\mathrel{+}=\frac{1}{\sqrt{d_k}}\mathrm{d}S^{\top}Q.
$$

全部按块累加、带因果掩码，得到 $O(T)$ 的显存复杂度（只存 $O$ 与 $L$，$S,P$ 在反向重算），代价是多一次 $S$ 的计算。FlashAttention-2/3 进一步改进了分块并行与 Hopper/Blackwell 的异步流水线；对应到 TPU，XLA 会自动融合出同类的分块注意力算子，不需要手写 CUDA。

### 6.4 融合核

- **RMSNorm**：前向一个核内完成平方均值、归一化与仿射；反向也合成一个核，避免反复读 $x$、$r$ 中间量；
- **SwiGLU**：$P\odot\mathrm{SiLU}(G)$ 与两个投影 GEMM 通过 epilogue/融合拼在一起，减少一轮显存往返；
- **fused softmax + cross-entropy**：策略头的 $z\in\mathbb{R}^{T_a\times|V|}$ 很大，标准做法是块内计算 logits、在线 LSE、立即求出 rowwise 损失，反向直接产出 $\mathrm{d}z=\exp(z-\mathrm{LSE})-e_y$（引理 2），无需物化完整的 softmax 概率矩阵；
- **embedding**：前向 gather，反向 scatter-add（原子加或分段规约）。权重绑定让 $\mathrm{d}W_E$ 收两路散射（4.3(b)）；
- **激活重计算**：$L_s=1024$、18+24 层，激活显存是主要开销；保存每个块的输入（或每若干层的边界），反向时重算块内激活，用约 $+30\%$ 的前向算力换大量显存。

### 6.5 梯度同步与内存切分

- **数据并行**：每张卡算本地平均梯度，用 ring all-reduce 求平均（5.6 节公式）。通信量是 $2(D-1)/D\times$ 梯度大小；实现上按桶（约 25 MB）边算边发，与反向计算重叠；
- **ZeRO/FSDP**：3B 参数下，fp32 主权重 12 GB、Adam 的 $m,v$ 各 12 GB、bf16 梯度 6 GB，单卡放不下。常见做法是参数/梯度/优化器状态分片（ZeRO-1/2/3，FSDP 用 all-gather + reduce-scatter 替代 all-reduce），本质是 6.5 节平均公式 + 分片存储；
- **梯度累积**：把全局 batch 4096 拆成若干 micro-batch，用 5.6 节的线性性累加梯度后再做一次 Adam 步，控制激活峰值。

### 6.6 TPU 侧：原系统实际走的路径

官方伪代码 `import jax, jax.numpy, optax`，论文 Fig. 3 标注 TPU v6e，训练总投入约 $8\times10^{4}$ TPU-days（等价 $4000$ 块 TPU 跑 $20$ 天）。对应的底层机制是：

- `jax.grad` 把损失函数 trace 成 jaxpr，再降到 XLA HLO，由 XLA 做算子融合（RMSNorm、SwiGLU、softmax-CE 等融合成少量大核）与内存规划；
- 矩阵乘降到 TPU 的 MXU（bf16 脉动阵列，fp32 累加），由 XLA 选择 tiling 与 layout，不需要手写内核；
- 多 TPU 用 SPMD/GSPMD 自动切分，梯度用 ICI 上的集合通信（all-reduce 等）完成，语义与 6.5 完全相同；
- Adam 状态是 pytree，`optax.apply_updates` 逐叶子相加，更新公式就是 5.2 的向量化形式。

因此“GPU 上用什么矩阵算法”的答案是：原论文没有用 GPU；若把同一公式搬到 GPU，就是 cuBLAS/CUTLASS 的 tensor-core GEMM、FlashAttention、融合核加 NCCL 集合通信，逐项对应关系如上。

### 6.7 一次 learner step 的完整公式链

把第 1–6 节串起来，主 RL 的一个参数更新是：

1. 从回放池采样 $n=4096$ 条已验证 $(s_i,a_i,G_i)$，按 $9:1$ 混入 Mathlib SFT 数据；
2. 前向：对每条样本编码状态得 $\hat H_i$，价值头得 $p_{v,i}$，解码器得 $z_{i,j}$；
3. 损失：

$$
\mathcal{L}(\theta)=\frac{1}{n}\sum_{i=1}^{n}
\Big[
-\frac{1}{|a_i|}\sum_{j=1}^{|a_i|}\log \mathrm{softmax}(z_{i,j})_{a_{i,j}}
-\lambda\log p_{v,i}\big(\mathrm{bin}(G_i)\big)
\Big],
\qquad \lambda=10^{-3};
$$

4. 反向：由第 4 节的引理与逐层公式得到 $g_k=\nabla_\theta\mathcal{L}$；多设备时先做 5.6 节的平均（等价于 all-reduce）；
5. 裁剪（重建）：$g_k\leftarrow g_k\min(1,c/\|g_k\|_2)$；
6. Adam：$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k$，$v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^{\odot2}$，$\widehat m_k=m_k/(1-\beta_1^k)$，$\widehat v_k=v_k/(1-\beta_2^k)$；
7. 更新：$\theta_k=\theta_{k-1}-\eta_k\widehat m_k/(\sqrt{\widehat v_k}+\varepsilon)$，其中 $\eta_k$ 由调度给出；
8. $\theta_k$ 周期性写入 checkpoint，供给 $3000$ 个 actor 做下一轮搜索；1M 步后得到主 RL 模型。

---

## 7. 事实校正：相对本目录前文与旧笔记的更新

拿到官方伪代码（Supplementary Data 1）与补充表（Tables 1–7）后，以下数值需要更新：

- 架构：编码器 $18$ 层、解码器 $24$ 层（不是旧推的 32/20）；$16$ 头、每头 $128$、$d=2048$（不是 $32\times64$）；FFN 加宽系数 $6$；价值 bin 数 $64$；每次扩展采样 $K=6$ 条战术（不是 $64$）。
- 损失：策略权重 $1.0$，价值权重 $10^{-3}$（官方表格与伪代码一致）；主 RL 的 SFT 混合 $10\%$，TTRL 为 $25\%$；主 RL batch 4096、回放池 $6\times10^{7}$、约 $10^{6}$ 步；TTRL 回放池 $2\times10^{7}$、约 $2.5\times10^{6}$ 步。
- 优化器：官方伪代码是 `optax.adam(lr)`（Adam，无 weight decay）；旧笔记 `06-main-rl.md` §4 的“(S) AdamW”是当时的猜测，应让位于伪代码。
- 学习率/调度/裁剪：论文与补充表均未公布，公开伪代码的 `lr=1.0` 是占位；复现时按 5.4 节重建并做扫描。
- 搜索超参（Supplementary Table 3 + 伪代码）：$K=6$、$C=0.01$、$\alpha=0.6$、$c_{\text{init}}=0.001$、$c_{\text{base}}=3200$、$\tau=200$、$c_{\text{AND}}=64$、未访问边惩罚 $32$、折扣 $0.99$；伪代码里“无合法动作”的价值为 $-40$。
- Matchmaker（Supplementary Table 7）：`trust_count = 8`、连续证明 `12` 次视为掌握、权重 $1.0/0.1/10^{-3}/0$（有趣/未决/已掌握/已证伪）、起始模拟数 $250$、失败乘子 $1.17$、上限 $16000$、$50\%$ 概率尝试证伪。
- 预训练/SFT（Supplementary Table 4）：batch 4096，预训练 $3\times10^{6}$ 步（对应编码器 $12$ T、解码器 $3$ T token），SFT $500$ 步；编码器输入 1024 token、解码器 256 token（预训练）/ 64 token（SFT）。

---

## 8. 材料索引

- 论文：T. Hubert et al., Nature 651, 607–613 (2026)，doi:10.1038/s41586-025-09833-y（开放获取）；全文抄录在 `/home/zhai/project/reap/alpha-proof-original-math-version.wsl-backup/refs-alpha-proof-paper-nature-2025.md`。
- 官方伪代码：Supplementary Data 1 `pseudocode.py`，本机副本 `/home/zhai/project/alphaproof-official-materials/pseudocode.py`。
- 官方补充表：Supplementary Tables 1–7，本机副本 `/home/zhai/project/alphaproof-official-materials/supplementary-information.pdf`（纯文本 `supplementary-information.txt`）。
- 官方示例仓库（已克隆到 `/home/zhai/project/`）：`formal-imo`（IMO 题目的 Lean 形式化）、`miniF2F`（Lean 4 版 miniF2F）、`alphaproof-nexus-results`（AlphaProof Nexus 生成的形式证明与自然语言伴随证明）。
- 本目录：《5-mcts-rl.md》《6-alpha-proof-mcts-rl.md》《8-6-q-transform-reason-1.md》《9-mainrl-ttrl-llm-value-head.md》；本文是其续篇，专注梯度、优化器与底层矩阵算法的完整公式化。
