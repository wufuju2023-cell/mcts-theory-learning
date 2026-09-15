Hoeffding 不等式是集中不等式的核心结果之一。下面从马尔可夫不等式出发，经 Chernoff 方法、Hoeffding 引理，完整推导出一般形式。

---

## 1. 陈述

设 $X_1,\dots,X_n$ 相互独立，且几乎必然有

$$
X_i\in [a_i,b_i],\quad i=1,\dots,n,
$$

其中 $a_i<b_i$。令

$$
S_n=\sum_{i=1}^n X_i,\qquad \mu=\mathbb{E}[S_n]=\sum_{i=1}^n \mathbb{E}[X_i].
$$

则对任意 $t>0$，

$$
\boxed{
\mathbb{P}(S_n-\mu\ge t)
\le
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
}
$$

同理，

$$
\mathbb{P}(S_n-\mu\le -t)
\le
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

双边版本为

$$
\boxed{
\mathbb{P}(|S_n-\mu|\ge t)
\le
2\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
}
$$

若考虑样本均值 $\bar X=\frac{1}{n} S_n$，则

$$
\mathbb{P}(\bar X-\mathbb{E}[\bar X]\ge \epsilon)
\le
\exp\left(
-\frac{2n^2\epsilon^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

特别地，若 $X_i\in[0,1]$，则

$$
\mathbb{P}(\bar X-\mathbb{E}[\bar X]\ge \epsilon)
\le e^{-2n\epsilon^2}.
$$

---

## 2. Chernoff 方法：从马尔可夫不等式开始

设 $Y$ 是随机变量，$t>0$，$\lambda>0$。由马尔可夫不等式，

$$
\mathbb{P}(Y\ge t)
=\mathbb{P}(e^{\lambda Y}\ge e^{\lambda t})
\le e^{-\lambda t}\,\mathbb{E}[e^{\lambda Y}].
$$

这就是 Chernoff 方法：用指数函数放大尾部事件，再控制矩母函数。

取

$$
Y=S_n-\mu=\sum_{i=1}^n (X_i-\mathbb{E}[X_i]).
$$

则

$$
\mathbb{P}(S_n-\mu\ge t)
\le
e^{-\lambda t}\,
\mathbb{E}\left[e^{\lambda(S_n-\mu)}\right].
$$

---

## 3. 独立性与矩母函数分解

令

$$
Y_i=X_i-\mathbb{E}[X_i].
$$

则 $\mathbb{E}[Y_i]=0$，且 $Y_i\in[a_i-\mathbb{E}[X_i],\,b_i-\mathbb{E}[X_i]]$。该区间的长度仍为 $b_i-a_i$。

由于 $X_1,\dots,X_n$ 独立，$Y_1,\dots,Y_n$ 也独立。因此

$$
\mathbb{E}\left[e^{\lambda(S_n-\mu)}\right]
=
\mathbb{E}\left[e^{\lambda\sum_{i=1}^n Y_i}\right]
=
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda Y_i}\right].
$$

所以只要对每个零均值有界随机变量 $Y_i$ 给出指数矩的上界即可。这正是 Hoeffding 引理。

---

## 4. Hoeffding 引理

**引理.** 设 $X$ 是随机变量，$\mathbb{E}[X]=0$，且 $X\in[a,b]$ 几乎必然，其中 $a<b$。则对任意 $\lambda\in\mathbb{R}$，

$$
\boxed{
\mathbb{E}[e^{\lambda X}]
\le
\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
}
$$

### 证明

若 $a=b$，则 $X=0$，不等式平凡。以下设 $a<b$。

因为 $\mathbb{E}[X]=0$ 且 $X\in[a,b]$，必有 $a\le 0\le b$。

函数 $x\mapsto e^{\lambda x}$ 是凸函数。对任意 $x\in[a,b]$，凸函数在区间上的值不超过两端点线性插值：

$$
e^{\lambda x}
\le
\frac{b-x}{b-a}e^{\lambda a}
+
\frac{x-a}{b-a}e^{\lambda b}.
$$

取 $x=X$ 并取期望，利用 $\mathbb{E}[X]=0$：

$$
\mathbb{E}[e^{\lambda X}]
\le
\frac{b}{b-a}e^{\lambda a}
-
\frac{a}{b-a}e^{\lambda b}.
$$

令

$$
p=\frac{-a}{b-a}\in[0,1],
$$

则

$$
1-p=\frac{b}{b-a}.
$$

再令

$$
u=\lambda(b-a).
$$

则

$$
a=-p(b-a),\qquad b=(1-p)(b-a).
$$

代入得

$$
\mathbb{E}[e^{\lambda X}]
\le
(1-p)e^{-pu}+p e^{(1-p)u}
=
e^{-pu}\bigl(1-p+p e^u\bigr).
$$

两边取对数：

$$
\ln \mathbb{E}[e^{\lambda X}]
\le
\ln\bigl(1-p+p e^u\bigr)-pu.
$$

定义

$$
L(u)=\ln\bigl(1-p+p e^u\bigr)-pu.
$$

则 $L(0)=0$。求导：

$$
L'(u)=\frac{p e^u}{1-p+p e^u}-p,
$$

所以 $L'(0)=0$。再求导：

$$
L''(u)
=
\frac{p(1-p)e^u}{(1-p+p e^u)^2}.
$$

令

$$
q=\frac{p e^u}{1-p+p e^u},
$$

则

$$
L''(u)=q(1-q)\le \frac{1}{4}.
$$

因此对 $u>0$，

$$
L(u)=\int_0^u\int_0^v L''(s)\,ds\,dv
\le
\int_0^u\int_0^v \frac{1}{4}\,ds\,dv
=
\frac{u^2}{8}.
$$

对 $u<0$ 同理可得相同上界。于是

$$
L(u)\le \frac{u^2}{8}.
$$

代回 $u=\lambda(b-a)$，得到

$$
\mathbb{E}[e^{\lambda X}]
\le
\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
$$

引理证毕。

---

## 5. 组合得到 Hoeffding 界

对每个 $Y_i=X_i-\mathbb{E}[X_i]$ 应用 Hoeffding 引理。注意 $Y_i$ 的区间长度为 $b_i-a_i$，且 $\mathbb{E}[Y_i]=0$。于是

$$
\mathbb{E}[e^{\lambda Y_i}]
\le
\exp\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right).
$$

因此

$$
\mathbb{E}\left[e^{\lambda(S_n-\mu)}\right]
=
\prod_{i=1}^n \mathbb{E}[e^{\lambda Y_i}]
\le
\prod_{i=1}^n
\exp\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right)
=
\exp\left(
\frac{\lambda^2}{8}\sum_{i=1}^n (b_i-a_i)^2
\right).
$$

记

$$
V=\sum_{i=1}^n (b_i-a_i)^2.
$$

则

$$
\mathbb{E}\left[e^{\lambda(S_n-\mu)}\right]
\le
e^{\lambda^2 V/8}.
$$

由 Chernoff 方法，

$$
\mathbb{P}(S_n-\mu\ge t)
\le
e^{-\lambda t}e^{\lambda^2 V/8}
=
\exp\left(-\lambda t+\frac{\lambda^2 V}{8}\right).
$$

---

## 6. 对 $\lambda$ 优化

上式对任意 $\lambda>0$ 成立。为得到最紧的界，令

$$
f(\lambda)=-\lambda t+\frac{\lambda^2 V}{8}.
$$

求导：

$$
f'(\lambda)=-t+\frac{\lambda V}{4}.
$$

令 $f'(\lambda)=0$，得

$$
\lambda=\frac{4t}{V}.
$$

代入：

$$
f\left(\frac{4t}{V}\right)
=
-\frac{4t^2}{V}
+
\frac{1}{8}\cdot \frac{16t^2}{V}
=
-\frac{4t^2}{V}+\frac{2t^2}{V}
=
-\frac{2t^2}{V}.
$$

因此

$$
\mathbb{P}(S_n-\mu\ge t)
\le
\exp\left(-\frac{2t^2}{V}\right)
=
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

对下尾，考虑 $-X_i$。它们仍独立，且 $-X_i\in[-b_i,-a_i]$，区间长度相同。因此

$$
\mathbb{P}(S_n-\mu\le -t)
\le
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

由并集界，

$$
\mathbb{P}(|S_n-\mu|\ge t)
\le
2\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

---

## 7. 样本均值形式

令 $\bar X=\frac{1}{n} S_n$，则 $\mathbb{E}[\bar X]=\frac{\mu}{n}$。对任意 $\epsilon>0$，

$$
\bar X-\mathbb{E}[\bar X]\ge \epsilon
\iff
S_n-\mu\ge n\epsilon.
$$

代入 $t=n\epsilon$，得到

$$
\mathbb{P}(\bar X-\mathbb{E}[\bar X]\ge \epsilon)
\le
\exp\left(
-\frac{2n^2\epsilon^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

若 $X_i\in[0,1]$，则 $b_i-a_i=1$，$V=n$，于是

$$
\boxed{
\mathbb{P}(\bar X-\mathbb{E}[\bar X]\ge \epsilon)
\le e^{-2n\epsilon^2}.
}
$$

这正是 UCB 推导中使用的形式。

---

## 8. 总结

推导链条为：

$$
\text{马尔可夫不等式}
\;\Longrightarrow\;
\text{Chernoff 指数方法}
\;\Longrightarrow\;
\text{独立性分解}
\;\Longrightarrow\;
\text{Hoeffding 引理}
\;\Longrightarrow\;
\text{优化 }\lambda
\;\Longrightarrow\;
\text{Hoeffding 不等式}.
$$

核心技巧是：

1. 用 $e^{\lambda Y}$ 放大尾部事件；
2. 用独立性把矩母函数拆成乘积；
3. 用凸性得到单个有界零均值变量的指数矩上界；
4. 对 $\lambda$ 求极小值，得到最优指数衰减率。

最终，Hoeffding 不等式给出了独立有界随机变量之和偏离期望的指数尾界。
