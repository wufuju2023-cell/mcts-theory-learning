马尔可夫不等式是很多集中不等式的起点。严格陈述如下。

---

## 马尔可夫不等式

设 $(\Omega,\mathcal{F},\mathbb{P})$ 是概率空间，$X$ 是非负随机变量，即

$$
X:\Omega\to[0,\infty],
$$

且 $a>0$。若 $\mathbb{E}[X]$ 存在，则

$$
\boxed{
\mathbb{P}(X\ge a)\le \frac{\mathbb{E}[X]}{a}.
}
$$

如果 $\mathbb{E}[X]=+\infty$，右边为 $+\infty$，不等式平凡成立。因此通常只需考虑 $\mathbb{E}[X]<+\infty$ 的情形。

---

## 证明：指示函数法

对任意 $\omega\in\Omega$，考虑指示函数

$$
\mathbf{1}_{\{X\ge a\}}(\omega)=
\begin{cases}
1,& X(\omega)\ge a,\\
0,& X(\omega)<a.
\end{cases}
$$

我们证明逐点不等式：

$$
a\,\mathbf{1}_{\{X(\omega)\ge a\}}\le X(\omega),
\quad \forall \omega\in\Omega.
$$

分两种情况：

1. 若 $X(\omega)\ge a$，则 $a\,\mathbf{1}_{\{X\ge a\}}=a\le X(\omega)$。
2. 若 $X(\omega)<a$，则 $a\,\mathbf{1}_{\{X\ge a\}}=0\le X(\omega)$，因为 $X$ 非负。

所以逐点不等式成立。两边取期望，由期望的单调性：

$$
a\,\mathbb{E}\bigl[\mathbf{1}_{\{X\ge a\}}\bigr]\le \mathbb{E}[X].
$$

而

$$
\mathbb{E}\bigl[\mathbf{1}_{\{X\ge a\}}\bigr]
=\mathbb{P}(X\ge a).
$$

因此

$$
a\,\mathbb{P}(X\ge a)\le \mathbb{E}[X].
$$

由于 $a>0$，两边除以 $a$，得到

$$
\boxed{
\mathbb{P}(X\ge a)\le \frac{\mathbb{E}[X]}{a}.
}
$$

证毕。

---

## 备注

同样的证明也可以写成

$$
\mathbb{P}(X>a)\le \frac{\mathbb{E}[X]}{a},
$$

因为 $\{X>a\}\subseteq \{X\ge a\}$，所以

$$
\mathbb{P}(X>a)\le \mathbb{P}(X\ge a)\le \frac{\mathbb{E}[X]}{a}.
$$

通常马尔可夫不等式写 $\ge$ 或 $>$ 都可以。

这个不等式的核心技巧是：
用截断变量 $a\mathbf{1}_{\{X\ge a\}}$ 从下方控制 $X$，然后取期望。

它也是切比雪夫不等式和 Hoeffding 不等式的起点。例如，对非负变量 $e^{tX}$ 使用马尔可夫不等式：

$$
\mathbb{P}(X\ge \varepsilon)
=\mathbb{P}(e^{tX}\ge e^{t\varepsilon})
\le e^{-t\varepsilon}\mathbb{E}[e^{tX}],
$$

这正是推导 Hoeffding 不等式的第一步。
