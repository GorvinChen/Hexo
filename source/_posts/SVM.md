---
title: SVM
tags: 
- ML
- SVM
categories: ML
mathjax: true
---

### 简介

支持向量机（support vector machine，SVM）是在分类与回归分析中的有监督学习模型。对于二分类问题，SVM将样本表示为空间中的点，通过超平面将样本正确分类且尽可能宽的间隔分开<!-- more -->。并通过将新的样本映射到同一空间，基于它们落在超平面的哪一侧来预测其所属的类别。除了进行线性分类之外，SVM还可以使用核方法有效地进行非线性分类，将其输入隐式映射到高维特征空间中。



### 线性可分

给定线性可分的训练数据集 $(x,y)$，通过间隔最大化得到分离的超平面为 $w^T x+b=0$，称之为 $(w, b)$，相应的分类决策函数为称之为线性可分支持向量机，如式(2.1)所示。
$$
f(x) = \mbox{sign}(w^Tx+b)\tag{2.1}
$$
​对于二分类问题，超平面要做的事情是将样本分开，而两类对应的函数值在超平面的符号正好相反，所以假设 $y$ 的标签为 $\pm1$，其优点是可以直观感受，并且推导的时候更加简便。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/mEGcBBf2fD.png?imageslim" style="zoom:65%" />



定义 $yf(x)$ 为函数间隔，当样本正确分类的时候 $yf(x) \geqslant 0$，但是我们不能最大化这个函数间隔，因为当超平面 $(w,b)$ 确定时，等比例放大 $w$ 和 $b$ 得到的函数间隔会以相同的比例增大，而超平面 $(kw,kb)$ 却还是之前的超平面。所以便引出了几何间隔
$$
 \frac{yf(x)}{\left \| w \right \|}\tag{2.2}
$$
其中 ${\left \| w \right \|}$ 表示 $L_2$ 范数。实际上这个几何间隔就是分类正确的点到直线距离公式。某条样本来说，该样本离超平面的间隔越大则分类的置信度越高，为了使分类的置信度尽量高，我们可以最大化几何间隔。然而对于样本空间中任意点 $x_i$ 到超平面都有一个几何间隔，那么我们到底最大化哪些点的几何间隔呢？

从下图中看到离超平面越远的点我们越不需要关注，因为它们的几何间隔肯定比正好在间隔边界上的点（背景色不透明的点）大。所以我们只需要最大化这些点的几何间隔就可以了，而这些在间隔边界上的点便被称为**支持向量**。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/mEGcBBf2fD.png?imageslim" style="zoom:65%" />

对于任意支持向量 $x_s$ 都有
$$
y_sf(x_s)=1\tag{2.3}
$$


#### 最大间隔

支持向量在边界上，所以支持向量正好满足 $w^Tx+b=\pm1$，所以我们可以直接最大化上图中的 $\gamma$ 得到如下的约束问题
$$
\begin{align*}
\max_{w,b}\quad&\frac{2}{ \left \| w \right \|} \\
\mbox{s.t.}\quad& y_i(w^Tx_i + b) \geqslant 1, \quad i=1,2,\cdots,n
\end{align*}\tag{2.4}
$$
而这个问题可以等价转化成如下的形式

$$
\begin{align*}
\min_{w,b}\quad&\frac{1}{2} \left \| w \right \|^2 \\
\mbox{s.t.}\quad& y_i(w^Tx_i + b) \geqslant 1, \quad i=1,2,\cdots,n
\end{align*}\tag{2.5}
$$


#### 问题求解

根据[KKT](https://scnico.github.io/2018/01/04/SVM/#%E4%B8%8D%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E9%97%AE%E9%A2%98)条件中不等式约束问题的求解，该问题可以转化成如下形式
$$
\begin{align*}

&L(w,b,\alpha)=\frac{1}{2} \left \| w \right \|^2  + \sum_i^n{\alpha_i [1-y_i(w^Tx_i+b)]} \\
&\mbox{s.t.}\quad \alpha_i \geq 0, \quad i=1,2,\cdots,n

\end{align*}\tag{2.6}
$$
根据[拉格朗日对偶性](https://scnico.github.io/2018/01/04/SVM/#%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E5%AF%B9%E5%81%B6%E6%80%A7)，且在本问题中[Slater](https://scnico.github.io/2018/01/04/SVM/#%E5%8E%9F%E5%A7%8B%E9%97%AE%E9%A2%98%E4%B8%8E%E5%AF%B9%E5%81%B6%E9%97%AE%E9%A2%98)条件成立，则可以将原始问题转化为对偶问题求解
$$
\min_{w, b}  \max_{\alpha} L(x, b, \alpha) \Rightarrow \max_{\alpha} \min_{w, b}   L(x, b, \alpha)\tag{2.7}
$$
转化为对偶问题的优点有两个：一是对偶问题往往更容易求解；二是自然引入核函数，进而推广到非线性分类。



##### 转化原始问题

首先固定 $\alpha$，求解 $\min_{w, b}   L(w, b)$，分别对 $w$ 和 $b $ 求导得到
$$
\begin{align*}
&\frac{\partial  L}{\partial  w}=0 \Rightarrow w=\sum_{i=1}^n \alpha_i y_i x_i\tag{2.8}\\
&\frac{\partial  L}{\partial  b}=0 \Rightarrow \sum_{i=1}^n \alpha_i y_i=0\tag{2.9}
\end{align*}
$$

将结果带入 $L(w,b,\alpha)​$ 消去 $w​$ 和 $b ​$ 后得到只含有变量 $\alpha​$ 的式子
$$
\begin{align*}
L(w,b,\alpha)&=\frac{1}{2} \left \| w \right \|^2  + \sum_i^n{\alpha_i [1-y_i(w^Tx_i+b)]} \\
&=\frac{1}{2}w^T\sum_{i=1}^n \alpha_i y_i x_i-w^T\sum_{i=1}^n \alpha_i y_i x_i - b\sum_{i=1}^n \alpha_i y_i + \sum_{i=1}^n \alpha_i \\
&= \sum_{i=1}^n \alpha_i - \frac{1}{2}(\sum_{i=1}^n \alpha_i y_i x_i )^T\sum_{i=1}^n \alpha_i y_i x_i \\
&=\sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\end{align*}\tag{2.10}
$$

该问题被转化成为
$$
\begin{align*}
\max_\alpha\quad & \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j\\

\mbox{s.t.}\quad
&\sum_{i=1}^n \alpha_i y_i = 0 \\
&\alpha_i \geq 0,\quad i=1,2,\cdots,n 
\end{align*}\tag{2.11}
$$
式(2.11)被称为原始问题的对偶问题，求对偶问题的解 $\alpha$ 带入到式(2.7)就能求出原始问题的解 $w$；再根据支持向量的定义即式(2.3)即可求出原始问题的解 $b$ 。



##### SMO求解对偶问题

SMO(Sequential Minimal Optimization)算法是一种启发式算法，主要用来求解[二次规划问题](https://zh.wikipedia.org/wiki/%E4%BA%8C%E6%AC%A1%E8%A7%84%E5%88%92)。其基本思想是若所有变量的解都满足该最优化问题的KKT条件，通过KKT条件便可以得到最优化问题的解。式(2.11)可以等价转化成如下形式

$$
\begin{align*}
\min_\alpha\quad & \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i=1}^n \alpha_i \tag{2.12}\\

\mbox{s.t.}\quad
&\sum_{i=1}^n \alpha_i y_i = 0 \tag{2.13}\\
&\alpha_i \geqslant 0, \quad i=1,2,\cdots,n\tag{2.14}
\end{align*}
$$
而该极小化问题正好是一个二次规划问题，我们可以通过SMO算法来求解。一般的优化算法通过梯度方法每次优化一个变量（固定其他变量）求解二次规划问题的最值，而上述问题由于限制条件(2.13)存在，若每次改变更新一个 $\alpha_i$ ，则更新后显然不满足该条件。所以SMO算法通过每次选择两个变量 $\alpha_i$ 和 $\alpha_j$ 进行优化（固定其他变量）。这样针对两个变量构建二次规划问题，这个子问题的解更接近于原始二次规划问题的解。

- 算法步骤

  while 未收敛 do

  - 通过启发式方法选取 $\alpha_i$ 和 $\alpha_j$ 
  - 固定 $\alpha_i$ 和 $\alpha_j$ 以外的参数，求解上述极小问题并更新 $\alpha_i$ 和 $\alpha_j$ 


出于连贯性这里先介绍怎么求解，然后再介绍怎么选择 $\alpha_i$ 和$\alpha_j$



###### 二次规划求解

不失一般性，假设选取 $\alpha_1$ 和 $\alpha_2$，固定剩余变量，省略不包含  $\alpha_1$ 和 $\alpha_2$ 的常数项，则问题可以写成
$$
\begin{alignat}{2}
\min_{\alpha_1,\alpha_2}\quad & W(\alpha_1,\alpha_2)=\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+\alpha_1\alpha_2K_{12}y_1y_2\\
&\quad\quad\quad\quad\quad\ \ \ -(\alpha_1+\alpha_2)+\alpha_1y_1v_1+\alpha_2y_2v_2\tag{2.15}
\\
\\
\mbox{s.t.}\quad
&\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^n\alpha_iy_i\tag{2.16} = \zeta\\
&0 \leqslant \alpha_i \leqslant C, \quad i=1,2\tag{2.17}
\end{alignat}
$$

式(2.16)中 $\zeta$ 表示一个常量，式(2.17)中引入了变量 $C$ ，这个变量称为惩罚系数，是一个大于0的数，具体的会在[软间隔](https://scnico.github.io/2018/01/04/SVM/#%E8%BD%AF%E9%97%B4%E9%9A%94)中讲到。本节末会分析当没有 $C$ 的情况。此外其中
$$
\begin{align*}
v_i&=\sum_{j=3}^n\alpha_jy_jK_{ij} \\
&=[\sum_{j=1}^n\alpha_jy_jK(x_i, x_j)+b] - [\sum_{j=1}^2\alpha_jy_jK(x_i, x_j)+b]\\
&={f}'(x) - [\sum_{j=1}^2\alpha_jy_jK(x_i, x_j)+b]\\
&={f}'(x)-(\alpha_1y_1K_{1i}+\alpha_2y_2K_{2i}+b)\tag{2.18}
\end{align*}
$$
上式给出了 ${f}'(x)$ 的定义，这样做除了可以简化公式外 ${f}'(x)$ 实际上跟我们最后求得的SVM分类函数类似，在计算时还可以用来引出误差（具体的在求解过程中）。下面假设更新前后 $\alpha_1$ 和 $\alpha_2$ 的值分别为 $\alpha_1^{old}，\alpha_2^{old}$ 和  $\alpha_1^{new}，\alpha_2^{new}$ 。由于 $y_i^2=1$，由式(2.16)可知

$$
\begin{align*}
& \alpha_1 =  y_1\zeta  -y_1y_2 \alpha_2 \tag{2.19}\\ 
\\
& \alpha_1^{old}y_1+\alpha_2^{old}y_2 = \alpha_1^{new}y_1+\alpha_2^{new}y_2 = y_1\zeta   \tag{2.20}
\end{align*}
$$
我们先不考虑式(2.17)的约束，将式(2.19)带入到式(2.15)可以得到
$$
\begin{align*}

 W(\alpha_2)=&\frac{1}{2}K_{11}( y_1\zeta  -y_1y_2 \alpha_2)^2+\frac{1}{2}K_{22}\alpha_2^2+( y_1\zeta  -y_1y_2 \alpha_2)\alpha_2K_{12}y_1y_2\\
 &-( y_1\zeta  -y_1y_2 \alpha_2+\alpha_2) +( y_1\zeta  -y_1y_2 \alpha_2)y_1v_1+\alpha_2y_2v_2

\end{align*} \tag{2.21}
$$
然后对 $\alpha_2​$ 求导另其为0可得
$$
\begin{align*}
\bigtriangledown W(\alpha_2)=&-y_1y_2K_{11}( y_1\zeta  -y_1y_2 \alpha_2) +K_{22}\alpha_2+y_2K_{12}\zeta -2K_{12}\alpha_2\\
&+y_1y_2-1-y_2v_1+y_2v_2=0

\end{align*} \tag{2.22}
$$
式(2.22) $\alpha_2$ 为更新后的 $\alpha_2$ ，所以我们可以将称为 ${\alpha_2^{new}}'$ ，再将其化简得
$$
\begin{align*}
(K_{11}+K_{22}-2K_{12}){\alpha_2^{new}}'&=y_2\zeta (K_{11}-K_{12})+y_2(v_1-v_2)-y_1y_2+1\\
&=y_2[\zeta (K_{11}-K_{12})+v_1-v_2-y_1+y_2]\\
\end{align*} \tag{2.23}
$$
然后将式(2.18)和式(2.20)带入上式替换掉 $\zeta$，$v_1$ 和 $v_2$ 化简我们可以得到等式右边为
$$
\begin{align*}
y_2[(K_{11}&+K_{22}-2K_{12})\alpha_2^{old}y_2+{f}'(x_1)-{f}'(x_2)+y_2-y_1]\\
&=(K_{11}+K_{22}-2K_{12})\alpha_2^{old}+y_2[({f}'(x_1)-y_1)-(f^*(x_2)-y_2)]\\
&=(K_{11}+K_{22}-2K_{12})\alpha_2^{old}+y_2(E_1-E_2) \tag{2.24}
\end{align*}
$$
其中 $E_i$ 表示误差令
$$
\eta=K_{11}+K_{22}-2K_{12}\tag{2.25}
$$
带入式(2.24)可得到
$$
{\alpha_2^{new}}'=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{\eta} \tag{2.26}
$$
再由式(2.20)我们可以得到
$$
\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new}) \tag{2.27}
$$
以上便是未考虑约束 $0 \leqslant \alpha_i \leqslant C, \  i=1,2​$ 的结果，当考虑该约束时 ${\alpha_2^{new}}'​$ 应该满足约束 $L \leq {\alpha_2^{new}}'\leq H​$ ，我们可以将约束用二维坐标系表示，根据 $\alpha_1y_1+\alpha_2y_2=\zeta ​$ 可得

若 $y_1 \neq y_2$
$$
\alpha_1^{new}-\alpha_2^{new}=\alpha_1^{old}-\alpha_2^{old}=k \tag{2.28}
$$
其中 $k​$ 是一个常量，为 $\zeta ​$ 或者 $-\zeta ​$，其在坐标系中的截距如下图所示

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/21Alf66cmE.png?imageslim" style="zoom:60%" />

则我们可以得到在这种情况下
$$
\begin{align*}
 &L = max(0,\alpha_2^{old}-\alpha_1^{old}) \\
 &H = min(C,C+\alpha_2^{old}-\alpha_1^{old}) \\
\end{align*}\tag{2.29}
$$
同理若 $y_1 = y_2​$
$$
\alpha_1^{new}+\alpha_2^{new}=\alpha_1^{old}+\alpha_2^{old}=k\tag{2.30}
$$
其在坐标系中的截距如下图所示

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/hg5ac8bF7f.png?imageslim" style="zoom:60%" />

在这种情况下
$$
\begin{align*}
 &L = max(0,\alpha_1^{old}+\alpha_2^{old}-C) \\
 &H = min(C,\alpha_1^{old}+\alpha_2^{old})\\
\end{align*} \tag{2.31}
$$
综上所述便得到了经剪辑（考虑约束 $0 \leqslant \alpha_i \leqslant C, \  i=1,2​$）后 $\alpha_2^{new}​$ 的解为
$$
\alpha_2^{new} = \begin{cases}
L,&\alpha_2^{new*} < L\\
\\
\alpha_2^{new*} ,&L \leq \alpha_2^{new*} \leq H\\
\\
H,&\alpha_2^{new*} >H\\
\end{cases}\tag{2.32}
$$
等等我们还忘记了一件事，这里的 $C​$ 是我们引入进来的，那么不加 $C​$ 会是什么情况呢。当不加限制 $C​$ 时即我们的 $C​$ 取无穷大，则我们可以得到下面的式子
$$
\begin{cases}
L=max(0,\alpha_2^{old}-\alpha_1^{old}),H =\infty&if\quad y_1 \neq y_2\\
\\
L=0,H=\alpha_1^{old}+\alpha_2^{old}&if\quad y_1=y_2\\

\end{cases}\tag{2.33}
$$
即当 $y_1 \neq y_2​$ 时
$$
\alpha_2^{new} = \begin{cases}
L,&\alpha_2^{new*} < L\\
\\
\alpha_2^{new*} ,&L \leq \alpha_2^{new*}\\

\end{cases}\tag{2.34}
$$
当 $y_1=y_2​$ 时
$$
\alpha_2^{new} = \begin{cases}
\alpha_2^{new*} ,&\alpha_2^{new*} \leq H\\
\\
H,&\alpha_2^{new*} >H\\
\end{cases}\tag{2.35}
$$



###### 计算偏移项b

在每次完成两个变量的优化后，都需要重新计算偏移项 $b$。对于任意支持向量 $(x_s,y_s)$ 根据KKT条件都会存在 $y_sf(x_s)=1$，即
$$
y_s(\sum_{i\in S}\alpha_iy_ix_i^Tx_s+b)=1\tag{2.36}
$$
其中 $S​$ 表示所有支持向量的集合，我们已经讨论过了，求得的分类函数 $f(x)​$ 中只需要计算支持向量的内积，而非支持向量的 $\alpha_i=0​$ 。理论上我们可以选取任意的支持向量获取 $b​$ 值，但实际上我们上是通过计算所有支持向量的均值。
$$
b=\frac{1}{|S|}\sum_{s\in S}(y_s-\sum_{i\in S}\alpha_iy_ix_i^Tx_s)\tag{2.37}
$$


###### 选择变量

SMO算法在选择变量优化时首先选择一个不满足KKT条件的变量，然后第二个变量的选择标准是希望可以使得第二个变量更新前后的变化足够大。

首先说什么是变化足够大，我们可以看 ${\alpha_2^{new}}'$ 的更新公式，我们发现该更新依赖于 $E_1-E_2$，为了加快计算可以简单的选择 $\alpha_2$ 使其对应的 $|E_1-E_2|$ 最大（假设 $\alpha_1$ 已选则 $E_1$ 已知）。就这样不断的在选定 $\alpha_1$ 的情况下选择更新 $\alpha_2$ 。

再说根据不满足KKT条件来选择第一个变量，求解该不等式约束问题是通过KKT条件，即要求
$$
\begin{cases}

\alpha_i(y_i f(x_i) - 1)=0\\
\\
\alpha_i \geqslant 0\\
\\
y_i f(x_i) - 1 \geqslant 0\

\end{cases}\tag{2.38}
$$
所以对于任意训练样本 $(x_i, y_i)​$，总有 $\alpha_i=0​$ 或者 $y_i f(x_i) = 1​$。

- $\alpha_i = 0$，即 $y_i f(x_i) \geq 1$，对应样本点位于最大间隔正确分类的一面，也可以理解为乘子为0，约束不起作用
- $\alpha_i > 0$，即 $y_i f(x_i) = 1$，对应样本点位于最大间隔的边界上，称之为支持向量

则违反KKT条件就是不满足上述条件，即选取的 $\alpha_i$ 满足
-  $y_i f(x_i) \geq 1$ 时 $\alpha_i > 0$，而原本 $\alpha_i = 0$
-  $y_i f(x_i) = 1$ 时 $\alpha_i = 0$，而原本 $\alpha_i>0$

当引入松弛变量后（建议先阅读软间隔）我们会发现此时 $\alpha_i$ 还可以等于 $C$。当 $\alpha_i=C$ 时可以得到 $y_if(x_i) \leq 1$（因为此时引入的松弛变量 $\xi_i \geq 0$）。所以我们还可以选取当 $y_if(x_i) \leq 1$ 时 $\alpha_i \leq C$ 而原本 $\alpha_i=C$ 的 $\alpha_i$。



#### 总结

通过求解对偶问题我们得到了最优解 $\alpha$，通过 $\alpha$ 我们就可以得到原始问题的解 $w$ 以及 $b$ 得到我们想要的分类函数了。其形式如下所示
$$
f(x)=\mbox{sign}(\sum_{i=1}^n\alpha_iy_ix_ix^T+b)\tag{2.39}
$$


### 线性不可分

我们之假定了训练样本是线性可分的，即存在某超平面可以将训练样本正确分类，而在实际的问题中，原始样本空间或许并不能由某个超平面正确分类。对于这样的问题可以通过将原始的样本空间映射到更高维的特征空间，使得样本在特征空间中线性可分。比如下图中的数据原本是线性不可分的，而我们从第三个维度却是线性可分的，而核函数就是在解决这个问题的。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/BFmbBbmK2J.png?imageslim" style="zoom:50%" />



#### 核函数

原始的用非线性可分的数据去训练一个线性分类器，通常的做法就是将原始样本空间映射到特征空间，然后在特征空间中训练线性分类器。而核函数的方法则是直接将特征空间的映射以及内积融合在一起，并且解决了映射函数维度爆炸的问题（多项式核中将会举一个简单的例子）。

首先定义 $\phi(x_i)$ 表示样本 $x_i$ 映射到特征空间的特征向量，定义核函数如下所示
$$
\kappa(x_i,x_j)=\left \langle \phi(x_i),\phi(x_j)  \right \rangle= \phi(x_i)^T\phi(x_j)\tag{3.1}
$$
考虑我们得到的线性可分函数
$$
f(x)=(\sum_{i=1}^n\alpha_iy_ix_i)^Tx+b\Rightarrow f(\phi(x))=\sum_{i=1}^n\alpha_iy_i\kappa(x_i,x)+b\tag{3.2}
$$

下面将介绍几个常见的核函数。



##### 多项式核

$$
\kappa(x_i,x_j)=(x_i^Tx_j+c)^d\tag{3.3}
$$

考虑简单的二维样本空间，$x_i=(a_1, a_2)^T​$，$x_j=(b_1,b_2)^T​$，并取 $d=2​$，分别取 $c=0​$ 和 $c=1​$
$$
\begin{align*}
&\left \langle x_i, x_j \right \rangle=a_1b_1+a_2b_2 \\
&(\left \langle x_i, x_j \right \rangle) ^2=a_1^2b_1^2+2a_1b_1a_2b_2+  a_2^2b_2 ^2\\
&(\left \langle x_i, x_j \right \rangle + 1) ^2=a_1^2b_1^2+2a_1b_1a_2b_2+  a_2^2b_2 ^2+2a_1b_1+2a_2b_2 +1
\end{align*}\tag{3.4}
$$
可以取映射 $\phi_1(x_i)=(a_1^2,\sqrt2a_1a_2,a_2^2)^T$ 则可以将原样本空间映射到三维空间，得到的结果与上式中第二个式子类似；

取映射 $\phi_2(x_i)=(a_1^2,\sqrt2a_1a_2,a_2^2,\sqrt2 a_1,\sqrt2 a_2, 1)^T$ 则可以将原样本空间映射到五维空间，得到的结果与上式中第三个式子类似。

考虑如果继续增大 $d$，那么如果我们通过原始的方法先映射到特征空间则需要映射到更多的维度，而如果用核函数则不存在这个维度爆炸的问题。

##### 高斯核

高斯核又称高斯径向基函数(radial basis function，RBF)，该核函数可以将原始的样本空间映射到无穷维，其形式如下所示
$$
\begin{align*}
\\
\kappa(x_i,x_j)&=\exp(-\frac{\left \|x_i-x_j  \right \|^2}{2\sigma^2}) \\
&= \exp(-\frac{\left \|x_i \right \|^2 + \left \|x_j  \right \|^2 - 2x_i^Tx_j}{2\sigma^2})\\
&= \exp(-\frac{\left \|x_i \right \|^2 }{2\sigma^2})\exp(-\frac{\left \|x_j \right \|^2 }{2\sigma^2})\exp(\frac{ 2x_i^Tx_j}{2\sigma^2})
\end{align*}\tag{3.5}
$$

根据指数函数的泰勒公式
$$
\exp(x)=\sum_{n=0}^\infty \frac{x^n}{n!}\tag{3.6}
$$
所以继续变换
$$
\kappa(x_i,x_j)== \exp(-\frac{\left \|x_i \right \|^2 }{2\sigma^2})\exp(-\frac{\left \|x_j \right \|^2 }{2\sigma^2}) \sum_{n=0}^\infty \frac{ (2x_i^Tx_j)^n}{2\sigma^2n!}\tag{3.7}
$$
根据泰勒展开式我们可以看到高斯核可以将数据映射到无穷维空间。

##### 其他核函数

当然除了多项式核以及高斯核以外还有很多核函数，下面将列出常见的核函数

- 线性核

  线性核实际上就是原始空间的内积，这个核主要是为了方便工程实现，不用将线性和非线性SVM分开，全部都用非线性来表示，只不过带入的核函数不同。
  $$
  \kappa(x_i,x_j)=x_i^Tx_j\tag{3.8}
  $$













- 拉普拉斯核

  $ \sigma > 0 $ 
  $$
  \kappa(x_i,x_j)=\exp(-\frac{\left \|x_i-x_j  \right \|}{\sigma})\tag{3.9}
  $$

- sigmoid核

  $\beta>0$，$\theta<0$

  $$
  \kappa(x_i,x_j)=\tanh(\beta x_i^Tx_j+\theta)
  $$












### 软间隔

在之前的讨论中，我们都假定了训练数据的样本空间或者特征空间是线性可分的，然而在实际任务中往往很难确定是否线性可分。缓解该问题的办法是允许支持向量机在一些样本上出错，所以便引入了软间隔，或者说松弛变量，即允许某些样本不满足约束 $y_if(x_i)\geq1$，我们对每条样本引入一个松弛变量 $\xi_i$ 将约束条件变为

$$
y_i(w^Tx_i+b)\geq1-\xi_i\tag{4.1}
$$
则相应的目标函数变为
$$
\begin{align*}
\min_{w,b}\quad&\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^n\xi_i\\
\mbox{s.t.}\quad& y_i(x^Tx_i + b) \geq 1-\xi_i, \quad i=1,2,\cdots,n\\
&\xi_i \geq 0, \quad i=1,2,\cdots,n
\end{align*}\tag{4.2}
$$
这里的 $C>0$ 称为惩罚参数，我们看到 $C$ 是所有 $\xi$ 之和的系数，而所有 $\xi$ 之和表明了样本越过边界的幅度以及有多少样本越过边界，所以 $C$ 值越大对误分类的惩罚越大，相反惩罚越小。同样的引入拉格朗日乘子得到拉格朗日函数
$$
\begin{align*}

L(w,b,\xi,\alpha,\mu)=&\frac{1}{2} \left \| w \right \|^2+ C\sum_{i=1}^n\xi_i\\ 
& + \sum_i^n{\alpha_i [1-y_i(w^Tx_i+b)-\xi_i]} -\sum_{i=1}^n\mu\xi_i\\
\end{align*}\tag{4.3}
$$
其中 $ \alpha_i \geq 0, \mu_i \geq 0$ 同理可以转化为对偶问题先求 $L$ 对 $w$，$b$ 以及 $\xi$  极小
$$
\begin{align*}
&\frac{\partial  L}{\partial  w}=0 \Rightarrow w=\sum_{i=1}^n \alpha_i y_i x_i\tag{4.4}\\
&\frac{\partial  L}{\partial  b}=0 \Rightarrow \sum_{i=1}^n \alpha_i y_i=0\tag{4.5}\\
&\frac{\partial  L}{\partial  \xi} = 0 \Rightarrow C-\alpha_i-\mu_i=0\tag{4.6}
\end{align*}
$$
带入 $L​$ 得到
$$
\begin{align*}
\min_{w,b,\xi}L(w,b,\xi,\alpha,\mu)=\sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\end{align*}\tag{4.7}
$$
再对其求极大得到对偶问题
$$
\begin{align*}
\max_\alpha\quad & \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j\tag{4.8}\\

\mbox{s.t.}\quad
&\sum_{i=1}^n \alpha_i y_i = 0 \tag{4.9}\\
&C-\alpha_i-\mu_i=0\tag{4.10}\\
&\alpha_i \geq 0,\quad i=1,2,\cdots,n \tag{4.11}\\
&\mu_i \geq 0,\quad i=1,2,\cdots,n \tag{4.12}
\end{align*}
$$
而式(4.10~12)可以消去 $\mu$ 得到
$$
0\leq\alpha_i\leq C\tag{4.13}
$$
这也是我们在上述SMO求解对偶问题一节中引入 $C$ 的原因，这样软间隔问题就和之前叙述的SMO问题形式类似了。但是KKT条件却如下所示
$$
\begin{cases}

\alpha_i(y_i f(x_i) - 1)=0\\
\\
y_i f(x_i) - 1 \geqslant 0\
\\

\\
\mu_i\xi_i=0\\
\\
\alpha_i \geqslant 0,\mu_i \geq 0,\xi_i \geqslant 0\\
\end{cases}\tag{4.14}
$$



### Hinge损失

回顾一下软间隔的目标函数
$$
\begin{align*}
\min_{w,b}\quad&\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^n\xi_i\\
\mbox{s.t.}\quad& y_i(w^Tx_i + b) \geq 1-\xi_i, \quad i=1,2,\cdots,n\\
&\xi_i \geq 0, \quad i=1,2,\cdots,n
\end{align*}\tag{5.1}
$$
对于该目标函数我们还可以理解为带 $L_2$ 正则的Hinge损失。我们先看一下Hinge函数的定义
$$
Hinge(x)=max(0, 1-x)\tag{5.2}
$$
其函数图像在坐标轴上如下图所示，因其形状像开门时 $135^{\circ}$ 的合页得名Hinge。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/3CjGJaBC3l.png?imageslim" style="zoom:40%" />



我们先定义包含损失函数的目标函数
$$
\min_{w,b}\quad\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^nL(f(x_i),y_i)\tag{5.3}
$$
一般的我们将上式第一项称为结构风险，用于描述得到的模型 $f(x)$ 的性质，而将第二项称为经验风险，用来描述模型与训练数据的契合度。而其中的系数 $C$ 是我们用来权衡结构风险和经验风险的系数。而在本题中经验风险我们用Hinge损失函数来表示
$$
L_{hinge}=max(0,1-y_if(x_i))\tag{5.4}
$$
所以目标函数变为
$$
\min_{w,b}\quad\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^nmax(0,1-y_if(x_i))\tag{5.5}
$$
由 $\xi_i\geq0$ 可以将上式重新写为
$$
\min_{w,b,\xi_i}\quad\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^n\xi_i \tag{5.6}
$$
而这和我们引入软件隔的目标函数正好一样。



### 支持向量回归
支持向量机通过最大间隔来得到分类模型，其将样本限定在决策边界的两边，$y_if(x_i)>0$ 才认为其正确分类，而支持向量回归（Support Vector Regression，SVR）恰好相反，SVR将样本点限制在决策边界之内，SVR认为只有在决策边界内的样本才是正确分类的样本，下图中的 $\varepsilon $ 为能够容忍的错误。
<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180115/fDH5e763Hf.png?imageslim" style="zoom:60%" />

所以SVR的目标函数为
$$
\min_{w,b}\quad\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^nL(f(x_i),y_i)\tag{6.1}
$$
其中
$$
L(f(x_i),y_i)=\begin{cases}
|f(x_i)-y_i|-\varepsilon, \\&|f(x_i)-y_i|>\varepsilon
\\
0,&otherwise
\end{cases}\tag{6.2}
$$
引入松弛变量 $\xi$，则目标函数变为
$$
\begin{align*}
\min_{w,b}\quad&\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^n\xi_i\\
\mbox{s.t.}\quad& w^Tx_i + b - y_i \leq \varepsilon + \xi_i, \quad i=1,2,\cdots,n\\
& y_i  - (w^Tx_i + b)  \leq \varepsilon - \xi_i, \quad i=1,2,\cdots,n\\
&\xi_i \geq 0, \quad i=1,2,\cdots,n
\end{align*}\tag{6.3}
$$

引入拉格朗日乘子 $\alpha​$，${\alpha}'​$ 和 $\mu​$，由拉格朗日乘子法可以得到拉格朗日函数
$$
\begin{align*}
L(w,b,\alpha,{\alpha}'_i,\mu,\xi)=&\frac{1}{2} \left \| w \right \|^2 + C\sum_{i=1}^n\xi_i + \sum_{i=1}^n{\alpha_i (f(x_i)-y_i-\varepsilon-\xi_i)}  \\
&+ \sum_{i=1}^n{\alpha}'_i (y_i-f(x_i)-\varepsilon+\xi_i)-\sum_{i=1}^n\mu_i\xi_i
\end{align*}\tag{6.4}
$$
分别对 $w​$，$b​$ 和 $\xi​$ 求偏导为零可以得到
$$
\begin{align*}
&\frac{\partial  L}{\partial  w}=0 \Rightarrow w=\sum_{i=1}^n ({\alpha}'_i-\alpha_i) x_i\tag{6.5}\\
&\frac{\partial  L}{\partial  b}=0 \Rightarrow \sum_{i=1}^n ({\alpha}'_i-\alpha_i) =0\tag{6.6}\\
&\frac{\partial  L}{\partial  \xi}=0 \Rightarrow C- \alpha_i+{\alpha}'_i-\mu_i=0\tag{6.7}\\

\end{align*}
$$
将式(6.5~7)带入式(6.4)得到（推导过程类似式(2.10)）
$$
\begin{align*}
L(\alpha,{\alpha}'_i,\mu)=&\sum_{i=1}^n\left[ \left({\alpha}'_i-\alpha_i\right)y_i -({\alpha}'_i+\alpha_i)\varepsilon\right]\\
&- \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n  ({\alpha}'_i-\alpha_i) ({\alpha}'_j-\alpha_j) x_i^T x_j \\
\end{align*}\tag{6.8}
$$
由式(6.7)可知 $\alpha_i-{\alpha} '_i \leq C​$，则原问题被转化成对偶问题
$$
\begin{align*}

 \max \quad & L(\alpha,{\alpha}'_i,\mu)=\sum_{i=1}^n\left[ \left({\alpha}'_i-\alpha_i\right)y_i -({\alpha}'_i+\alpha_i)\varepsilon\right]\\
&\quad\quad\quad\quad\quad\ \ \ - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n  ({\alpha}'_i-\alpha_i) ({\alpha}'_j-\alpha_j) x_i^T x_j  \\

\mbox{s.t.} \quad
    &\sum_{i=1}^n ({\alpha}'_i-\alpha_i) =0 \\
    & \alpha_i,{\alpha}'_i \geq 0, \quad i=1,2,\cdots,n\\
    &\alpha_i-{\alpha} '_i \leq C \quad i=1,2,\cdots,n
\end{align*}\tag{6.9}
$$
而不等式约束问题需要满足KKT条件即
$$
\begin{cases}
\alpha_i(f(x_i)-y_i-\varepsilon -\xi_i)=0\\
\\
{\alpha} '_i(f(x_i)-y_i-\varepsilon+\xi_i)=0\\
\\
\mu_i\xi_i=0\\
\\
 \alpha_i,{\alpha}'_i,\mu_i \geq 0
\end{cases}\tag{6.10}
$$
分析KKT条件可以看出当 $(f(x_i)-y_i-\varepsilon -\xi_i)=0$ 时，样本点落在下侧间隔带上，当 $f(x_i)-y_i-\varepsilon+\xi_i)=0$ 时样本点落在上侧间隔带上。由于样本点不可能同时落在下侧和上侧间隔带上，所以 $\alpha_i$ 和 ${\alpha} '_i$ 中必有一个为0。与求解SVM类似求解该对偶问题即可求解出 $\alpha$ 以及 ${\alpha} '_i$，然后再求出 $w$ 及 $b$。此外我们根据式(6.5)可以得到SVR的解的形式为
$$
f(x)=\sum_{i=1}^n({\alpha} '_i-\alpha_i)x_i^Tx+b\tag{6.11}
$$
可以看到SVR也可以表示成核函数的形式。 



### 拉格朗日对偶性

#### 原始问题

假设 $f(x)$，$c_i(x)$，$h_j(x)$ 是定义在 $\mathbf{R}^n$上的连续可微函数，考虑约束的最优化问题（原始问题）
$$
\begin{align*}
\min_{x \in \mathbf{R}^n}\quad & f(x) \tag{7.1}\\

\mbox{s.t.}\quad
&c_i(x) \leqslant 0, \quad  i=1,2,\cdots,k \tag{7.2}\\
&h_j(x) = 0, \quad j=1,2,\cdots,l\tag{7.3}
\end{align*}
$$
引进拉格朗日函数
$$
L(x, \alpha, \beta) = f(x) + \sum_{i=1}^{k} \alpha_i c_i(x) + \sum_{j=1}^{l} \beta_j h_j(x),\ \alpha_i \geqslant 0\tag{7.4}
$$
将关于 $x$ 的函数 $\Theta_P(x)$ ：
$$
\Theta_P(x) = \max_{\alpha, \beta: \alpha_i \geqslant 0} L(x, \alpha, \beta)\tag{7.5}
$$
假设给定某个 $x$，若 $x$ 违反原始问题的约束条件，即存在某个约束 $i$ 使得 $c_i(x) > 0$ 或者某个 $j$ 使得 $h_j(x)=0$，我们可以分别让 $\alpha_i \rightarrow +\infty$，$\beta_j h_j(x) \rightarrow +\infty$ 而其余的 $\alpha$ 和 $\beta$ 都取为 0，则 $\Theta_P(x)=+\infty$；相反如 $x$ 满足约束条件，则 $\Theta_P(x)=f(x)$。所以对于
$$
\min_{x} \Theta_P(x) = \min_{x}  \max_{\alpha, \beta: \alpha_i \geqslant 0} L(x, \alpha, \beta)\tag{7.6}
$$
来说其与原始我们想要求解的问题是等价的，原始问题便被我们转化成了极小极大问题。我们定义 $p^*$ 为原始问题的最优解。



#### 对偶问题

有时候解原始问题是非常困难的，所以通常可以通过求解对偶问题而得到原始问题的解。对偶问题的定义如下所示：
$$
\max_{\alpha, \beta: \alpha_i \geqslant 0} \Theta_D(x) = \max_{\alpha, \beta: \alpha_i \geqslant 0} \min_{x}   L(x, \alpha, \beta)\tag{7.7}
$$
这样问题就由原来的极小极大问题转化成了极大极小问题。我们定义 $\max_{\alpha, \beta: \alpha_i \geqslant 0} \Theta_D(x)​$ 为原始问题的对偶问题，$d^*​$为对偶问题的最优解。



#### 原始问题与对偶问题

若原始问题和对偶问题都有最优解，则
$$
\Theta_D(\alpha,\beta)=\min_xL(x, \alpha, \beta) \leqslant L(x, \alpha, \beta) \leqslant \max_{\alpha, \beta: \alpha_i \geqslant 0} L(x, \alpha, \beta) = \Theta_P(x)\tag{7.8}
$$
所以有
$$
\max_{\alpha, \beta: \alpha_i \geqslant 0} \Theta_D(x) \leqslant \min_x \Theta_P(x)\tag{7.9}
$$
则
$$
d^* =  \max_{\alpha, \beta: \alpha_i \geqslant 0} \min_{x}   L(x, \alpha, \beta) \leqslant \min_{x}  \max_{\alpha, \beta: \alpha_i \geqslant 0} L(x, \alpha, \beta) = p^*\tag{7.10}
$$

弱对偶性(weak duality)
$$
d^* \leqslant p^*\tag{7.11}
$$
强对偶性(strong duality)
$$
d^* = p^*\tag{7.12}
$$
那么什么时候强对偶性成立呢，这就要说到**Slater**条件

- 主问题为凸优化问题，$f(x)$ 是凸函数，$g_i(x)$ 是[凸函数](https://zh.wikipedia.org/wiki/%E5%87%B8%E5%87%BD%E6%95%B0)，$h_j(x)$ 是[仿射函数](https://baike.baidu.com/item/%E4%BB%BF%E5%B0%84%E5%87%BD%E6%95%B0/9276178?fr=aladdin)，且其可行域中至少有一点使不等式约束严格成立。

在满足Slater条件下，强对偶性成立，通过求解对偶问题，主问题也可以解决了。可以看到我们通过求解对偶问题的最优解 $\alpha$ 和 $\beta$，然后在通过 $\alpha$ 和 $\beta$ 能得到原始问题的最优解 $x$。



#### KKT条件

**KKT** (Karush-Kuhn-Tucker) 条件是判断在优化问题中(约束条件含有等式约束以及不等式约束) $x^*$ 和 $\alpha^*,\beta^*$ 分别是原始问题和对偶问题的解的充分必要条件。
$$
\begin{cases}
\bigtriangledown_x L(x^*,\alpha^*,\beta^*) = 0\\
\\
\bigtriangledown_\alpha L(x^*,\alpha^*,\beta^*)  = 0\\
\\
\bigtriangledown_\beta L(x^*,\alpha^*,\beta^*)  = 0\\
\\
\alpha_ic_i(x)=0\\
\\
c_i(x)\leq0\\
\\
\alpha_i \geqslant 0\\
\\
h_j(x)=0
\end{cases}\tag{7.13}
$$

##### 等式约束问题

为了便于理解我尽量使用上述提到的符号，在讨论KKT条件前我们先讨论等式约束问题：
$$
\begin{align*}
\min_{x}\quad & f(x)\tag{7.14} \\

\mbox{s.t.}\quad
&h_j(x) = 0, \quad  j=1,2,\cdots,l\tag{7.15}
\end{align*}
$$
对于上述问题我们可以引入拉格朗日乘子 $\beta$ 求解
$$
L(x, \beta) = f(x) + \sum_{j=1}^l \beta_j h_j(x)\tag{7.16}
$$
然后通过 $L(x, \beta)$ 分别对 $x$ 和 $\beta$ 求导，令其为 0 可得到可能极值点，具体是否为极值点需要根据实际情况验证。

##### 不等式约束问题

$$
\begin{align*}
\min_{x}\quad & f(x) \tag{7.17}\\

\mbox{s.t.}\quad
&c_i(x) \leq 0, \quad  i=1,2,\cdots,k\tag{7.18}
\end{align*}
$$

对于上述不等式约束问题我们可以通过引入松弛变量 $ \xi_i^2$ 将其转化为等式约束问题。
$$
\begin{align*}
&h_i(x, c_i) = c_i(x) + \xi_i^2 = 0\\
\end{align*}\tag{7.19}
$$
同样引入拉格朗日乘子 $\alpha$，构建拉格朗日函数
$$
L(x, \alpha, c) = f(x) + \sum_{i=1}^k \alpha_i (c_i(x) + \xi_i^2)\tag{7.18}
$$
同样对其求导可得
$$
\begin{cases}
\frac{\partial L}{\partial x} = \bigtriangledown f(x)+ \sum_{i=1}^k \alpha_i \bigtriangledown c_i(x) = 0\\
\\
\frac{\partial L}{\partial \alpha_i} = c_i(x) + \xi_i^2=0\\
\\
\frac{\partial L}{\partial \xi_i} = 2 \alpha_i \xi_i = 0\\
\\
\alpha_i \geqslant 0
\end{cases} \tag{7.19}
$$
观察上式第三个等式，可分为两种情况：

- $\alpha_i = 0$，$\xi_i \neq 0$，即乘子为 0，约束 $c_i(x)$ 不起作用，且根据第二个式子 $c_i(x) < 0$
- $\alpha_i \geqslant 0$，$\xi_i = 0$，即松弛变量为 0，约束 $c_i(x)$ 起作用，且根据第二个式子$c_i(x) = 0$ 

所以第三个等式和第二个等式可以合并成一个等式，合并后的式子便成为不等式约束优化问题的KKT条件
$$
\begin{cases}
\bigtriangledown f(x)+ \sum_{i=1}^k \alpha_i \bigtriangledown c_i(x) = 0\\
\\
\alpha_ic_i(x)=0\\
\\
\alpha_i \geqslant 0
\end{cases}\tag{7.20}
$$
当然我们还得讲讲为什么要求 $\alpha\geq0$。设 ${x}^*$ 是 $x$ 领域上的点，即
$$
{x}^* = x + \Delta x\tag{7.21}
$$
则根据 $c_i(x)\leq0$ 以及 $c_i(x + \Delta x)=c_i(x)+\bigtriangledown c_i(x)\Delta x $可得
$$
\bigtriangledown c_i(x)\Delta x \leq 0\tag{7.22}
$$
此外由于
$$
f(x + \Delta x)=f(x)+\bigtriangledown f(x)\Delta x\tag{7.23}
$$
根据上式可得为了在点 $x​$ 处取得极小值 $\bigtriangledown f(x)\Delta x \geq 0​$，而我们KKT条件中第一个式子可得
$$
\bigtriangledown f(x)\Delta x = -\sum_{i=1}^n\alpha_i\bigtriangledown c_i(x)\Delta x\tag{7.24}
$$
所以 $\alpha_i \geq 0$。

综上所述，当既有等式约束又有不等式约束时就有了我们之前所说的KTT条件（式7.13）。



### Code

```python
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error

def process_data(dataset):
    feat = dataset.data
    target = dataset.target

    # print dataset info
    print('feature shape: ', feat.shape)
    print('feature names: ', dataset.feature_names)
    print('target shape: ', target.shape)
    if hasattr(dataset, 'target_names'):
        print('target names: ', dataset.target_names)

    trainX, testX, trainY, testY = train_test_split(feat, target, test_size=0.2, random_state=6)
    train_data = {'feat': trainX, 'target': trainY}
    test_data = {'feat': testX, 'target': testY}
    print('train feature shape: ', trainX.shape)
    print('test feature shape: ', testX.shape)

    return train_data, test_data

if __name__ == '__main__':
    # support vector classification
    iris = load_iris()
    iris_train, iris_test = process_data(breast_cancer)

    svc_model = SVC()
    svc_model.fit(iris_train['feat'], iris_train['target'])
    svc_predict = svc_model.predict(iris_test['feat'])
    svc_score = accuracy_score(svc_predict, iris_test['target'])
    print('svc accuracy: ', svc_score)

    # support vector regression
    boston = load_boston()
    boston_train, boston_test = process_data(boston)

    svr_model = SVR()
    svr_model.fit(boston_train['feat'], boston_train['target'])
    svr_predict = svr_model.predict(boston_test['feat'])
    svr_score = mean_squared_error(svr_predict, boston_test['target'])
    print('svr accuracy: ', svr_score)
```



### 参考

[统计学习方法-李航](https://book.douban.com/subject/10590856/)

[机器学习-周志华](https://book.douban.com/subject/26708119/)

[支持向量机系列-pluskid](http://blog.pluskid.org/?page_id=683)

[支持向量机通俗导论（理解SVM的三层境界）-July](http://blog.csdn.net/v_july_v/article/details/7624837)

[浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)

[SVM](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)

[凸函数](https://zh.wikipedia.org/wiki/%E5%87%B8%E5%87%BD%E6%95%B0)

[仿射函数](https://baike.baidu.com/item/%E4%BB%BF%E5%B0%84%E5%87%BD%E6%95%B0/9276178?fr=aladdin)

[二次规划问题](https://zh.wikipedia.org/wiki/%E4%BA%8C%E6%AC%A1%E8%A7%84%E5%88%92)

































