[TOC]

# Gated recurrent unit

**GRU是循环神经网络的一个变种，相比于LSTM而言，他的结构更加简洁，但是他的效果却不比LSTM差**，LSTM 通过门控机制使循环神经网络不仅能记忆过去的信息，同时还能选择性地忘记一些不重要的信息而对长期语境等关系进行建模，而 GRU 基于这样的想法在保留长期序列信息下减少梯度消失问题。**根据 paper作者Cho, et al. 在 2014 年的介绍，GRU 旨在解决标准 RNN 中出现的梯度消失问题。**GRU 也可以被视为 LSTM 的变体，因为它们基础的理念都是相似的，且在某些情况能产生同样出色的结果。

## GRU 原理

GRU 背后的原理与 LSTM 非常相似，**即用门控机制控制输入、记忆等信息而在当前时间步做出预测**。以下为重要表达公式：
$$
\begin{aligned} z &=\sigma\left(x_{t} U^{z}+s_{t-1} W^{z}\right) \\ r &=\sigma\left(x_{t} U^{r}+s_{t-1} W^{r}\right) \\ h &=\tanh \left(x_{t} U^{h}+\left(s_{t-1} \circ r\right) W^{h}\right) \\ s_{t} &=(1-z) \circ h+z \circ s_{t-1} \end{aligned}
$$
GRU 有两个有两个门，==即一个重置门（reset gate）==和**一个更新门（update gate）**。从直观上来说，**重置门决定了如何将新的输入信息与前面的记忆相结合**，更新门定义了前面记忆保存到当前时间步的量。如果我们将重置门设置为 1，更新门设置为 0，那么我们将再次获得标准 RNN 模型。使用门控机制学习长期依赖关系的基本思想和 LSTM 一致，但还是有一些关键区别：

- GRU 有两个门（重置门与更新门），而 LSTM 有三个门（输入门、遗忘门和输出门）。 

- **GRU 并不会控制并保留内部记忆（c_t），且没有 LSTM 中的输出门。** 
- ==LSTM 中的输入与遗忘门对应于 GRU 的更新门==，重置门直接作用于前面的隐藏状态。 
- 在计算输出时并不应用二阶非线性。 

GRU单元如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504223605.png)

GRU 是标准循环神经网络的改进版，但到底是什么令它如此高效与特殊？

为了解决标准 RNN 的梯度消失问题，GRU 使用了更新门（update gate）与重置门（reset gate）。**基本上，这两个门控向量决定了哪些信息最终能作为门控循环单元的输出。**这两个门控机制的特殊之处在于，它们能够保存长期序列中的信息，且不会随时间而清除或因为与预测不相关而移除。为了解释这个过程的机制，我们将具体讨论以下循环网络中的单元传递过程。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504224339.png)

单个门控GRU单元如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504224545.png)

在详细介绍GRU单元的工作原理之前，我们先做以下的符号规定：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504224748.png)

### 更新门

在时间步 t，我们首先需要使用以下公式计算更新门 $z_t$：
$$
z_{t}=\sigma\left(W^{(z)} x_{t}+U^{(z)} h_{t-1}\right)
$$
其中$W^{(z)}、U^{(z)}$分别为更新门的权重矩阵，$x_{t}、h_{t-1}$分别为$t$时刻的输入与$t-1$时刻的输出信息。更新门将这两部分信息相加并投入到 Sigmoid 激活函数中，因此将激活结果压缩到 0 到 1 之间。以下是更新门在整个单元的位置与表示方法：

⭐注⭐：一次矩阵乘法就是一次线性变换过程。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504232802.png)

**更新门帮助模型决定到底要将多少过去的信息传递到未来**，或==到底前一时间步和当前时间步的信息有多少是需要继续传递的。==这一点非常强大，因为模型能决定从过去复制所有的信息以减少梯度消失的风险。我们随后会讨论更新门的使用方法，现在只需要记住更新门$z_t$ 的计算公式就行。 

### 重置门

本质上来说，重置门主要决定了到底有多少过去的信息需要遗忘，我们可以使用以下表达式计算：
$$
r_{t}=\sigma\left(W^{(r)} x_{t}+U^{(r)} h_{t-1}\right)
$$
其中$W^{(r)}、U^{(r)}$代表权重矩阵，$x_{t}、h_{t-1}$分别为$t$时刻的输入与$t-1$时刻的输出信息。重置门将这两部分信息相加并投入到 Sigmoid 激活函数中，因此将激活结果压缩到 0 到 1 之间。下图展示了该运算过程的表示方法：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504233929.png)

### 当前记忆内容 

当前的记忆内容由重置门控制，**在重置门的使用中，新的记忆内容将使用重置门储存过去相关的信息（将新的信息和前面的记忆内容相结合）**，它的计算表达式为：
$$
h_{t}^{\prime}=\tanh \left(W x_{t}+r_{t} \odot U h_{t-1}\right)
$$
注意：$\odot$ 代表Hadamard 乘积，也就是矩阵对应元素直接相乘

输入 $x_t$ （新内容）与上一时间步信息 $h_{(t-1)}$ （前面的记忆内容）先经过一个线性变换，即分别右乘矩阵 $W $和 $U$。计算重置门 $r_t$ 与 $Uh_{(t-1)}$ 的 Hadamard 乘积，即 $r_t$ 与 $Uh_{(t-1)}$ 的对应元素乘积。**因为重置门是一个由 0 到 1 组成的向量，它会衡量门控开启的大小。例如某个元素对应的门控值为 0，那么它就代表这个元素的信息完全被遗忘掉。**==该 Hadamard 乘积将确定所要保留与遗忘的以前信息。==

将这两部分的计算结果相加再投入双曲正切激活函数中。该计算过程可表示为：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190504235954.png)

### 当前时间步的最终记忆

当前时间$t$时的最终记忆由更新门控确定，**网络计算 $h_t$时，该向量将保留当前单元的信息并传递到下一个单元中**，在这个过程中，我们需要使用更新门，它决定了当前记忆内容 $h'_t$ 和前一时间步 $h_{(t-1)}$ 记忆组合后的最终记忆$h_{t}$。这一过程可以表示为：
$$
h_{t}=z_{t} \odot h_{t-1}+\left(1-z_{t}\right) \odot h_{t}^{\prime}
$$
$z_t**$ 为更新门的激活结果，它同样以门控的形式控制了信息的流入**。$z_t$ 与 $h_{(t-1)}$ 的 Hadamard 乘积表示前一时间步保留到最终记忆的信息，该信==息加上==当前记忆保留至最终记忆的信息就等于最终门控循环单元输出的内容。以上表达式可以展示为：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190505001520.png)

以上四步计算便是GRU是通过更新门$z$与重置门$r$存储并过滤信息的全部过程，我**们可以知道门控循环单元不会随时间而清除以前的信息，它会保留相关的信息并传递到下一个单元，因此它利用全部信息而避免了梯度消失问题。**

## GRU训练过程

[GRU训练过程](https://ilewseu.github.io/2018/01/20/GRU%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/)

## GRU paper

## GRU实现

[pytorch实现GRU](https://github.com/WebLearning17/SequencePrediction)

[tensorflow实现GRU](https://blog.csdn.net/h8832077/article/details/80400462)

## 参考文献

[GRU详解](https://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/78888834)

[GRU门控的简明解释](https://zhuanlan.zhihu.com/p/28297161)

[GRU演示](https://www.itcodemonkey.com/article/9970.html)