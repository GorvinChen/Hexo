---
title: BP神经网络
tags: 
- ML
categories: ML
mathjax: true
---

### 概述

本篇文章是看了BP神经网络后而写，里面包含了BP神经网络的具体推导，我在看BP神经网络的时候产生的一些疑问，以及具体的matlab调用。<!-- more -->

### 神经元模型

神经网络中最基本的成分是神经元，在神经网络中每个神经元与其他神经元相连接，神经元通过连接的路径向下一个神经元发送信息。

### 感知机模型

感知机由两层神经元组成，如下图所示，其中$x_i$表示神经元的第$i$个输入，$w_i$表示第$i$个输入链接的权重，$\theta$表示神经元的阈值，达到阈值的神经元 ，$f$表示神经元的激励函数，$O(Output)$表示神经元的输出。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/29lBFc7hI4.jpg?imageslim" style="zoom:90%">

对于感知机来说，激励函数为阶跃函数
$$
f=sgn(x)=\begin{cases}
 & \text 1,\ \  x\geq0 \\ 
 & \text 0,\ \ x<0
\end{cases}
$$
学习规则，即权重更新为$w_i=w_i+\Delta w_i$，$\Delta w_i = \eta(T-O)x_i$，其中$\eta$为学习速率，$T$为目标输出或者说真实值。

为什么这么更新权重有效呢（为什么要在$\Delta w_i$后面乘以一个$x_i$呢）：学习速率$\eta$一般情况下是大于0的，若$T-O >0$，那么我们就需要让感知机的输出$O$更接近真实值$T$，所以我们需要增大$O$，而对于来$x_i$说，如果$x_i>0$，则$x_i$前面的系数$w_i$就需要增大，即$\Delta w_i$应该大于0，这时候我们通过公式$\Delta w_i = \eta (T-O)x_i$发现$\Delta w_i>0$，$w_i$根据这个公式更新，符合我们的需求；同理$x_i < 0$时也成立。

而后来人们发现感知机模型处理的问题实在是有限，连基本的“异或”问题都解决不了。于是乎便产生了多层神经网络。多层神经网络由输入层，隐藏层和输出层组成，每一层都包含多个神经元，多层神经网络的学习能力大大超过了单层的感知机模型，而要训练多层网络，显然感知机的学习规则过于简单是用不上了，于是便出现了经典的误差逆向传播算法（BackPropagation，简称BP）。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/jjhdaa1HCG.jpg?imageslim" style="zoom:90%">





### BP神经网络模型
#### 前向传播

首先在BP神经网络模型中我们讨论前向传播，对于一个$L$层的BP神经网络，其输入层为第$1$层，输出层为第$1$层，隐藏层为第$2$层到第$L-1$层。对于输出层的第$k$个神经元，其输出为$O_k^L$，下标$k$表示第$k$个，上标$L$表示层数，输入为$I_k^L$。第$l$层包含的神经元个数用$|l|$表示。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/0ecd7D4bkm.jpg?imageslim" style="zoom:90%">

在下面的表述中为了描述的更加清楚我将不使用阈值 $\theta$，即第 $l$ 层的第 $j$ 个神经元的输入 $I_j^l$ 与第 $l-1$ 个神经元的输出关系如下所示，其中 $w_{ji}^{l-1}$ 表示第 $l$ 层第$j$个神经元与第$l-1$层第$j$个神经元的链接权重。
$$
I_j^l=\sum_{i=1}^{|l-1|}w_{ji}^{l-1}·O_i^{l-1}
$$
而输出$O_j^l= f(I_j^l)$，$f$为激励函数。

##### 激活函数的选择

一般来说在神经网络中我们选择的激励函数$f​$是指函数$sigmoid​$，即$f=\frac{1}{1+e^{-x}}​$，在本篇文章中也会使用$sigmoid​$函数，那为什么我们需要选择函数呢：因为如果我们不选用激活函数或者选用线性的激活函数，你会发现，我去不管多少层网络，我们最后得到的都是一个线性函数，而使用非线性的$sigmoid​$以后，我们的网络就可以拟合非线性的函数了。当然$sigmoid​$函数在求导的时候也有一个非常有趣的现象，待会这个求导在下面会用到，具体可以自己推导下，$f'=f·(1-f)​$，即$\frac{\partial O_j^l}{\partial I_j^l}=O_j^l·(1-O_j^l)​$。当然还有其他很多激励函数，例如$tanh(x)= 2sigmoid(2x)-1​$，$ReLu = max(0,x)​$具体的大家可以看这篇[博客](http://blog.csdn.net/cyh_24/article/details/50593400)

#### 反向传播

OK，重点来了，反向传播，反向传播是干嘛的呢，反向传播就是通过最后的误差来更新权重的。简单的来说，就像一个公司搞了一波大事情，结果取得的效果并不好，那怎么办呢，董事长就说了总经理你去给我改，总经理就给下面的部门经理说你去给我改，... ，一直到普通的员工都得改，对应于我们的神经网络就是从输出层一直改到输入层。O.O讲正事，反向传播会用到一个非常重要的法则，那就是梯度下降法则。

梯度下降法则就不过多的叙述了，简单的说就是你更新权重不能更新呀，而按照梯度的方向更新权重能够更快的收敛。梯度下降的公式如下所示：
$$
E=\frac{1}{2}\sum_{k=1}^{|L|}(T_k-O_k^L)^2
$$

$$
w^l_{ji}=w^l_{ji}+\Delta w^l_{ji} \ \ \ \ \ \ \Delta w^l_{ji}=-\eta·\frac{\partial E}{\partial w^l_{ji}}
$$

那么为什么梯度下降要用负梯度方向呢：大前提，学习速率$\eta >0$，对于误差$E$来说，要是它对$w_{ji}^l$的偏导$\frac{\partial E}{\partial w_{ji}^l}$，那就说明误差随着权重的增大而增大，那我们就需要去通过减小权重来减小误差，所以$\Delta w_{ji}^{l} $就应该小于0，耶，你会发现这时候的$\Delta w_{ji}^{l} $公式中的负号就起作用了， 它正好使得$\Delta w_{ji}^{l} <0$。当然我们如果要求某某某的最大值，也可以使用正梯度。

直观上来看在神经网络中的每个神经元的权重更新都是通过误差对其的偏导来更新的，在这里我们需要引入新变量梯度$\delta $，对于第$l$层第$j$个神经元的梯度$\delta _{j}^{l} $，我们定义其值为$\delta _{j}^{l} =-\frac{\partial E}{\partial I_{j}^{l} } $，好了那接下来我们来看看权重到底是怎么更新的。

##### 输出层权重更新

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/GaaE84hHlc.jpg?imageslim" style="zoom:90%">

首先我们来看输出层，即第$L$层，由于权重$w_{kj}^{L-1} $只对输出层的第$k个单$元产生影响，即只影响$O_{k}^{L} $，所以我们直接求导就可以了。在求导过程中采用链式求导法则，具体如下

根据$I_{j}^{l} $的公式我们可以得到$I_{k}^{L}$以及$\frac{\partial I_{k}^{L}}{\partial w_{kj}^{L-1}} $
$$
-\frac{\partial E}{\partial w_{kj}^{L-1}}=\frac{\partial E}{\partial I_k^L}·\frac{\partial I_k^L }{\partial w_{kj}^{L-1}}=\delta_k^L·O_j^{L-1}
$$
根据$sigmoid$函数求导公式我们可以得到$\frac{\partial O_{k}^{L}}{\partial I_{k}^{L}} $，而误差$E$如下所示，所以我们可以得到$\frac{\partial E}{\partial O_{k}^{L}} $，最后便得到了第$L层的每个$神经单元的梯度项$\delta _{k}^{L} $
$$
E= \frac{1}{2}\sum_{k=1}^{|L|}(T_k-O_k^L)^2
$$

$$
\delta_k^L=-\frac{\partial E }{\partial I_k^L}=-\frac{\partial E}{\partial O_k^L}·\frac{\partial O_k^L}{\partial I_k^L}=(T_k-O_k^L)·O_k^L(1-O_k^L)
$$

所以便得到了输出层上一层与输出层的链接权重$w_{kj}^{L-1} $更新公式
$$
\Delta w_{kj}^{L-1}=\eta ·\delta_k^L·O_k^{L-1}
$$
##### 隐藏层权重更新

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/GaaE84hHlc.jpg?imageslim" style="zoom:90%">

接下来是隐藏层，这里对$\Delta w_{kj}^{L-2} $进行推导，其他隐藏层以及输入层类似，显然$w_{kj}^{L-2} $虽然只影响第$L-1$层的第$j个$单元，但$L-1$层的第$j$个单元却链接了输出层的所有单元，所以$w_{kj}^{L-2}$对输出层的每一个单元都产生了影响。我们需要在求导的时它对输出层单元的每个误差的影响进行叠加，与输出层类似部分就不再累述。
$$
-\frac{\partial E}{\partial w_{ji}^{L-2}}=\frac{\partial E}{\partial I_j^L-1}·\frac{\partial I_j^L-1 }{\partial w_{jt}^{L-2}}=\delta_j^{L-1}·O_i^{L-2}
$$
对于第$L-1$层的第$j$个神经单元的梯度项$\delta _{j}^{L-1} $来说，由于这个神经单元链接了输出层的每个单元，所以我们需要对输出层$\left| L \right| $个单元求和，于是便得到了梯度项$\delta _{j}^{L-1} $的计算公式
$$
\delta_j^{L-1}=-\frac{\partial E }{\partial I_j^{L-1}}=\sum_{k=1}^{|L|}-\frac{\partial E}{\partial I_k^L}·\frac{\partial I_k^L}{\partial O_j^{L-1}}·\frac{\partial O_j^{L-1}}{\partial I_j^{L-1}}=\sum_{k=1}^{|L|}\delta_k^L·w_{kj}^{L-1}·O_j^{L-1}(1-O_j^{L-1})
$$
所以便得到了$L-2$层与$L-1层$的链接权重$w_{kj}^{L-1} 更新$公式
$$
\Delta w_{kj}^{L-2}=\eta ·\delta_j^{L-1}·O_i^{L-2}
$$
而对于其他隐层以及输入层，我们发现可以把梯度项进行推广就得到了第$l$层第$i$个神经单元的梯度计算公式以及权重更新公式
$$
\delta_i^l=O_i^l(1-O_i^l)·\sum_{j=1}^{|l+1|}w_{ji}^l·\delta_j^{l+1}
$$

$$
\Delta w_{ji}^l=\eta ·\delta_j^{l+1}·O_i^j
$$

对于推导过程，简单的来说就是链式求导，由上面公式我们可以看出反向传播也是求导，只是对于隐层以及输入层来说，它们所产生的误差在传播过程中能够影响每一个最终的输出层单元，所以我们需要对其产生的梯度求和。

怎么记忆呢：反向传播的前提是前向传播，而前向传播的公式相信大家很容易都能记住，而反向传播其实就是对前向传播的公式链式求导，输出层$E$对$O^{L} $求导，$O^{L}$对$I^{L}$ 求导，$I^{L} $再对$w$求导（其中$E$，$O^{L}$ ，$I^L$ 表示其计算公式），计算隐层的时候就需要$E$对$O^{L} $求导，$O^{L}$对$I^{L} $求导，而$I^{L}$再对上一层的$O^{L-1} 求导$，$O^{L-1}$ 再对$I^{L-1} $求导，这时候如果还要继续往前推的话就需要继续往前对$O^{L-2} $求导，若只求这一层了就只需要$I^{L-1}$对$w$导了。

注意在求导过程中考虑$w$对输出层的影响来判断什么时候需要求和操作。

#### 算法步骤

1. 创建神经网络

   隐藏层一般来说为一层，若为多层时，每层单元数默认相同，一般单元数大于输入层单元数，输入层单元数为特征数量，输出层单元数为目标数量。

   隐藏层单元数$M$满足$M>log_2m$，其中$m$代表训练样本数

2. 初始化网络权重
   随机初始化一个很小的值，不能全置为0，全置为0后每次更新权重都会是一样的结果。

3. 将$m$个训练样本循环输入模型中进行训练
   a) 前向传播计算每个神经单元的输出
   b) 计算输出层梯度项 $\delta _{k}^{L}= O_{k}^{L}\cdot (1-O_{k}^{L})\cdot (T_{k}-O_{k}^{L} )$
   c) 计算隐藏层梯度项 $\delta _{i}^{l}= O_{i}^{l}\cdot (1-O_{i}^{l})\cdot \sum_{j=1}^{|l+1|}w_{ji}^l \cdot \delta_j^{l+1}$

4. 更新每个神经单元的权重$w_{ji}^{l} =w_{ji}^{l} +\Delta w_{ji}^{l} $，$\Delta w_{ji}^{l} =\eta \cdot \delta _{j}^{l+1}\cdot O_{i}^{l}$

#### 算法缺陷

1. 参数多，较复杂

2. 训练速度慢，容易陷入局部最优
   对于这个问题可以增加冲量项来解决，公式如下，其中$\alpha$为冲量，$0\leq \alpha  < 1，$$n$表示第$n$次更新
   $$
   \Delta w_{ji}^{l}=\eta ·\delta_j^{l+1}·O_i^{l}+\alpha\Delta w_{ji}^l(n-1)
   $$
   那么为什么这样做有效呢：若本次更新权重与上次更新方向相反，假设$\Delta w_{ji}^{l}(n)>0$，而$\Delta w_{ji}^{l}(n-1)<0$，即梯度的方向变了，说明错过了最小值，那么增加冲量就会让本次更新的步长减小点；若相同时可以增大步长，可以加快收敛。同样也可能使迭代过程中直接越过局部最优解。

3. 对初始权重敏感

   初始权重的不同会导致我们每次得到的结果有可能不同，在matlab中你可能发现我们每次训练的神经网络结果都不一样

#### Matlab工具包调用

本来想放自己的代码的，但看了看不忍直视，= . =，准备到时候改改在上传，然后想到我们既然都有了轮子，我们一般需要用到BP神经网络的时候就直接用轮子就可以了，当然主要还是时间紧外加我懒，具体的算法实现有时间再来更新吧。先给大家Matlab的干货，这边用的是[鸢尾花](http://pan.baidu.com/s/1sl6dFPV)的数据集来做BP神经网络分类。

```matlab
%———————————数据预处理—————————————
IrisData= importdata('IrisData.txt');
rows = size(IrisData, 1);

%由于数据的最后分类标签为1,2,3所以将其分别转化成1,0,0; 0,1,0; 0,0,1
Target = zeros(rows, 3);
for i = 1 : rows
Target(i, IrisData(i, 5)) = 1 ;
end
IrisData= [IrisData(:, 1:4), Target];

%选取第一类的前35个样本，第二类的前35个样本
%第三类的前35个样本作为训练集，剩下部分作为测试集
Train = [IrisData(1:35, :); IrisData(51:85, :); IrisData(101:135, :)];
Test = [IrisData(36:50, :); IrisData(86:100, :); IrisData(136:150, :)];

%input是输入的特征，而output是输入
%这里需要转置，因为在matlab的BP神经网络中数据是按列输入进去的
inputTrain = Train(:, 1:4)';
outputTrain = Train(:, 5:7)';
inputTest = Test(:, 1:4)';
outputTest = Test(:, 5:7)';

%数据缩放
%premnmx函数将数据的区间缩放到[-1, 1]
[inputTrain, minI, maxI] = premnmx(inputTrain); 
%tramnmx函数将数据的按照minI和maxI映射
inputTest = tramnmx (inputTest, minI, maxI); 
%———————————拟合神经网络—————————————
%创建神经网络，这边matlab中有新旧两版网络，具体可以百度一下
%[10, 3]表示创建了隐藏层为10个单元，输出层为3个单元的神经网络
%logsig为sigmoid函数，purelin表示线性函数，对应为神经网络每层选的激励函数
%traingdx表示带动量的梯度下降算法
net = newff(minmax(inputTrain), [10 3], {'logsig', 'purelin'}, 'traingdx');

%设置训练参数
net.trainparam.show = 50; %每隔50次训练显示一次收敛曲线的变化
net.trainparam.epochs = 500; %迭代次数为500次
net.trainparam.goal = 0.01; %目标梯度0.01
net.trainParam.lr = 0.01; %学习速率0.01

%训练网络
net = train(net, inputTrain, outputTrain);
%预测结果
predict = sim(net , inputTest); 

%———————————统计分类的正确率—————————————
[rows, cols] = size(predict);
rightNum = 0;
[predictValue, predictLabs] = max(predict);
[Value, Labs] = max(outputTest);
for i = 1:cols
if(predictLabs(i) == Labs(i)) 
rightNum = rightNum + 1; 
end
end
rightRate = rightNum/cols;
sprintf('神经网络对鸢尾花的类别分类正确率为：%3.2f%%',100 * rightRate)
```

在代码的注解中相信描述的已经很清楚了，然后这个训练好的net是可以保存下来的，当然也可以查看每个神经元的权重，参考[MATLAB 神经网络训练参数解释](http://link.zhihu.com/?target=http%3A//blog.sina.com.cn/s/blog_5c9288aa0101gsu2.html)。

好了，我们来看看结果，在对神经网络进行训练的时候会出现下面这个可视化的模型窗口，有四个部分，第一个部分表示神经网络结构，第二个部分表示训练的算法，这里是带动量的梯度下降，Preformance是误差指标，这里是均方差(MSE)；第三部分是训练进度，Epoch表示迭代次数，Time训练时间，Performance模型评价，Gradient梯度，Validation Checks为交叉验证，后面有数值的表示达到那个值就停止训练，都可以设置；第四部分是画图，就是可视化训练的结果。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/AfF2KDgeGC.jpg?imageslim" style="zoom:90%">

好了我们来看看我们训练的正确率到底是多少呢

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/7a2Bjj4lKC.jpg?imageslim" style="zoom:90%">



没错100%，当然你得到的结果不一样也是正常的，因为初始权重的不同会导致结果不一样。比如我就又跑了个97.78%出来了。

<img src="http://p2l71rzd4.bkt.clouddn.com/blog-image/180309/CJG0mg7bi9.jpg?imageslim" style="zoom:90%">


嗯，全部记完了，自己忘了的时候也可以来瞧一眼，当然时间有点仓促，里面可能还存在问题，最关键的是希望各位大神多提意见，当然有对本文内容不懂的也可以私信我！
