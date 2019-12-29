[TOC]

#  Recurrent neural network

**循环神经网络** recurrent neural network（**RNN**）是神经网络的一种。单纯的RNN因为无法处理随着递归，权重指数级爆炸或梯度消失的问题（Vanishing gradient problem），难以捕捉长期时间关联；而结合不同的**LSTM**可以很好解决这个问题。类似卷积神经网络（CNN）被专门开发出来处理图像问题，RNN专门用于处理序列数据。

## Introduce

让我们从一个问题开始，你能理解下面这句英文的意思吗？“working love learning we on deep”，答案显然是无法理解。那么下面这个句子呢？“We love working on deep learning”，整个句子的意思通顺了！我想说的是，**一些简单的词序混乱就可以使整个句子不通顺。**那么，我们能期待传统神经网络使语句变得通顺吗？不能！如果人类的大脑都感到困惑，我认为传统神经网络很难解决这类问题。

在日常生活中有许多这样的问题，当顺序被打乱时，它们会被完全打乱。例如，

- 我们之前看到的语言——单词的顺序定义了它们的意义
- 时间序列数据——时间定义了事件的发生
- 基因组序列数据——每个序列都有不同的含义

有很多这样的情况，**序列的信息决定事件本身**。如果我们试图使用这类数据得到有用的输出，就需要一个这样的网络：能够访问一些关于数据的**先前知识（prior knowledge）**。==也就是说我们需要学习序列的信息！==以便完全理解这些数据。**常见的一些序列关系如词语的顺序、时间序列、基因序列等等。**

在深入了解循环神经网络的细节之前**，让我们考虑一下我们是否真的需要一个专门处理序列信息的网络。**还有，我们可以使用这样的网络实现什么任务。 递归神经网络的优点在于其**应用的多样性**。当我们使用RNN时，它有强大的处理各种输入和输出类型的能力。看下面的例子。

- **情感分析(Sentiment Classification)** 最简单的情景是将一个句子划分为正负两种情绪
- **图像标注(Image Captioning)** 假设我们有一个图片，我们需要一个对该图片的文本描述。这里我们的输入是单一的图像，输出是一系列或序列单词。
- **机器翻译(Language Translation)** – 这里假设我们想将英文翻译为法语. 每种语言都有自己的语义，对同一句话有不同的长度。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190404215321.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190404215751.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190404215854.png)

通过上面的三个应用场景，我们可以看出RNNs可以很好地将输入映射到不同类型、长度的输出，并根据实际应用泛化。

## 原始RNN

RNN最先用于自然语言处理领域，比如，RNN可以为**语言模型**来建模(==也就是基于神经网络的分布式表示)==，**语言模型**就是这样的东西：给定一个一句话前面的部分，预测接下来最有可能的一个词是什么。**语言模型**是对一种语言的特征进行建模，它有很多很多用处。比如在语音转文本(STT)的应用中，声学模型输出的结果，往往是若干个可能的候选词，这时候就需要**语言模型**来从这些候选词中选择一个最可能的。当然，它同样也可以用在图像到文本的识别中(OCR)。

在未使用RNN时最常用的语言模型是N-gram，N是一个自然数，比如2或者3。它的含义是，假设一个词出现的概率只与前面N个词相关，但是这个模型无法对长依赖进行很好的建模，并且模型的大小和N的关系是指数级的，**4-Gram模型就会占用海量的存储空间。**提出RNNs后，理论上我们可以对向前(向后)看任意个词。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190404222800.png)

上图为最基本的循环神经网络的架构，它由一个输入层、一个隐藏层和一个输出层构成。所谓循环神经网络的精髓就是在那个循环箭头$W$，如果把上面有$W$的那个带箭头的圈去掉，它就变成了最普通的**全连接神经网络**。$x$是一个向量，它表示**输入层**的值；$s$是一个向量，它表示**隐藏层**的值（这里隐藏层面画了一个节点，你也可以想象这一层其实是多个节点，节点数与向量$s$的维度相同）；$U$是输入层到隐藏层的**权重矩阵**。$o$也是一个向量，它表示**输出层**的值；$V$是隐藏层到输出层的**权重矩阵**。==那么，现在我们来看看$W$是什么。==**循环神经网络**的**隐藏层**的值$s$不仅仅取决于当前这次的输入$x$，还取决于上一次**隐藏层**的值$s$。**权重矩阵**$W$就是**隐藏层**上一次的值作为这一次的输入的权重。如果我们把上面的图展开，**循环神经网络**也可以画成下面这个样子：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190404230650.png)

网络在$t$时刻接收到输入$x_t$之后，隐藏层的值是$s_t$，输出值是$o_t$。关键一点是，$s_{t}$的值不仅仅取决于$x_{t}$，还取决于$s_{t-1}$上一时刻隐藏层的值。我们可以用下面的公式来表示**循环神经网络**的计算方法：(这里为了简化，我们省略了偏置项b)
$$
o_{t}=g(Vs_{t})      \tag{1}
$$

$$
s_{t} = f(Ux_{t}+Ws_{t-1}) \tag{2}
$$

**式1**是**输出层**的计算公式，输出层是一个**全连接层**(也就是它的每个节点都和隐藏层的每个节点相连)。$V$是输出层的**权重矩阵**，$g$是**激活函数**。式2是隐藏层的计算公式，它是**循环层**。$U$是输入$x$的权重矩阵，$W$是上一次的值作为这一次的输入的**权重矩阵**，$f$是**激活函数**。

从上面的公式我们可以看出，**循环层**和**全连接层**的区别就是**循环层**多了一个**权重矩阵** $W$。如果反复将式2带入式1，我们可以得到以下结论。
$$
\begin{eqnarray*}
o_{t} &=& g(Vs_{t}) \tag{3} \\
 &=& Vf(Ux_{t}+Ws_{t-1}) \tag{4} \\
 &=& Vf(Ux_{t}+Wf(Ux_{t-1}+Ws_{t-2})) \tag{5} \\
 &=& Vf(Ux_{t}+Wf(Ux_{t-1}+Wf(Ux_{t-2}+Ws_{t-3}))) \tag{6} \\
 &=& Vf(Ux_{t}+Wf(Ux_{t-1}+Wf(Ux_{t-2}+Wf(Ux_{t-3}+...)))) \tag{7}
\end{eqnarray*}
$$
从上面可以看出，循环神经网络的输出值$o_{t}$，是受到前面历次输入值$x_{t},x_{t-1},x_{t-2},...$的影响。这就是循环神经网络的关键，它使得循环神经网络可以忘前看任意多个**输入值**，换句话说，==这样就学习了序列信息。【通过权重矩阵W，学习了序列信息】==

## 激活函数

RNN一般选择ReLu作为激活函数：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405155849.png)
$$
f(x) = max(0,x)
$$
还有一些常用的变种ReLu，选择这类ReLu作为激活函数的原因主要因为是在训练过程中避免出现梯度饱和情况导致训练无法进行，同时有因为是线性，训练速度会比较快。

## 损失函数

损失函数通过对比输出 $y_t$ 和目标 $z_t$ 之间的差距而评估了神经网络的性能，它可以形式化表达为：
$$
\mathcal L(y,z) = \sum_{t=1}^{T} \mathcal L_{t}(y_{t},z_{t})
$$
该表达式**对每一个时间步上的损失进行求和而得出最终的损失函数**。损失函数的挑选一般与具体问题相关，一般比较流行的损失函数包括预测实数值的欧几里德距离和 Hamming 距离，和用于分类问题 [13] 的交叉熵损失函数。

## 训练RNN

有效地训练 RNN 一直是重要的话题，该问题的难点在于网络中**难以控制的权重初始化**和==最小化训练损失的优化算法。==这两个问题很大程度上是由==网络参数之间的关系==和**隐藏状态的时间动态**而引起。

目前很多方法所展现的**关注点很大程度上都在于降低训练算法的复杂度，且加速损失函数的收敛。**然而，<font color=red>这样的算法通常需要大量的迭代来训练模型</font>。训练 RNN 的方法包括多表格随机搜索、时间加权的伪牛顿优化算法、梯度下降、扩展 kalman 滤波（EKF）[15]、Hessian-free、期望最大化（EM）[16]、逼近的 Levenberg-Marquardt [17] 和全局优化算法。

[RNN前向传播和BPTT实例](https://zhuanlan.zhihu.com/p/32755043)

[训练RNN的trick](https://www.zhihu.com/question/57828011)

[支持并行计算的SRNN](https://www.infoq.cn/article/sliced-recurrent-neural-networks)

## 双向RNN

传统的 RNN 在训练过程中只考虑数据的过去状态，但是对于很多情况而言考虑未来的状态也非常有必要。比如对于**语言模型**来说，很多时候光看前面的词是不够的，**比如下面这句话：**

- 我的手机坏了，我打算____一部新手机。

可以想象，如果我们只看横线前面的词，手机坏了，那么我是打算修一修？换一部新的？还是大哭一场？这些都是无法确定的。==但如果我们也看到了横线后面的词是『一部新手机』，那么，横线上的词填『买』的概率就大得多了。==

双向 RNN（BRNN）利用了过去和未来的所有可用输入序列评估输出向量。其中，需要用一个 RNN 以**正向时间**方向处理从开始到结束的序列，以及用另一个 RNN 处理**以反向时间方向处理**从开始到结束的序列，其结构如图所示

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405163116.png)

从上图可以看出，**双向卷积神经网络**的隐藏层要保存两个值，一个$A$参与正向计算，另一个值$A'$参与反向计算。最终的输出值$y_{2}$取决于$A_{2},A_{2}^{'}$的和。其计算方法为：
$$
y_{2} = g(VA_{2}+V^{'}A_{2}^{'})
$$
其中$A_{2},A_{2}^{'}$则分别计算为：
$$
\begin{eqnarray*}
A_{2} &=& f(WA_{1}+Ux_{2}) \tag{8} \\
A_{2}^{'} &=& f(W^{'}A_{3}^{'}+U^{'}x_{2}) \tag{9}
\end{eqnarray*}
$$
现在，我们已经可以看出一般的规律：正向计算时，隐藏层的值$s_{t}$与前一时刻的值$s_{t-1}$有关；反向计算时，隐藏层的值$s_{t}{'}$与后一个时刻$s_{t+1}{'}$有关；最终的输出取决于正向和反向计算的**加和**。现在，我们仿照**式1**和**式2**，写出双向循环神经网络的计算方法：
$$
\begin{eqnarray*}
o_{t} &=& g(Vs_{t}+V^{'}s^{'}) \tag{10} \\
s_{t} &=& f(Ux_{t}+Ws_{t-1}) \tag{11} \\
s_{t}{'} &=& f(U^{'}x_{t}+W^{'}s^{'}_{s+1}) \tag{12} \\
\end{eqnarray*}
$$
从上面我们可以看出，**正向计算和反向计算不共享权重**，也就是说$U$和$U^{'}$、$W$和$W^{'}$、$V$和$V^{'}$都是不同的权重矩阵。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405203509.png)

## 深度循环神经网络

前面介绍RNN只有一个隐藏层，我们可以堆叠多个隐藏层，这样就得到了深度循环神经网络。其结构如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405203752.png)
$$
\begin{eqnarray*}
o_{t} &=& g(V^{(i)}s_{t}^{(i)}+V^{'(i)}s^{'(i)}_{t}) \tag{13} \\
s_{i} &=& g(U^{(i)}s^{(i-1)}_{t}+W^{(i)}s_{t-1}) \tag{14} \\
s^{'}_{i} &=& f(U^{'(i)}s^{'(i-1)}+W^{'(i)}s^{'}_{t+1}) \tag{15} \\
s^{(1)} &=& f(U^{(1)}x_{t}+W^{(1)}s_{t-1}) \tag{16} \\
s^{'(1)} &=& f(U^{'(1)}x_{t}+W^{'(1)}s^{'}_{t+1}) \tag{17}
\end{eqnarray*}
$$

## BPTT算法

BPTT算法是针对**循环层**的训练算法，它的基本原理和BP算法是一样的，同样包含三个步骤：

- 前向计算每个神经元的输出值
- 反向计算每个神经元的误差项$\delta_{j}$的值，它的误差函数$E$对神经元$j$的加权输入$net_{j}$的偏导数
- 计算每个权重的梯度

**最后再用随机梯度下降算法更新权重**，循环层如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405222507.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405222749.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405223114.png)

### 前向计算

前向计算通过前向传播公式如下计算：
$$
s_{t} = f(Ux_{t}+Ws_{t-1})
$$
我们假设输入向量x的维度是m，输出向量s的维度是n，则矩阵U的维度是，矩阵W的维度是。下面是上式展开成矩阵的样子，看起来更直观一些：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405224654.png)

前向过程和普通的神经网络训练并没有什么差别

### 误差项的计算         

BTPP算法将第l层t时刻的**误差项**$\delta^{l}_{t}$值沿两个方向传播，**一个方向是其传递到上一层网络，得到$\delta^{l-1}_{t}$，这部分只和权重矩阵U有关**；==另一个是方向是将其沿时间线传递到初始时刻，得到$\delta^{l}_{1}$，这部分只和权重矩阵W有关。==

我们用向量$net_{t}$表示神经元在t时刻的**加权输入**，因为
$$
\begin{eqnarray*}
net_{t} &=& Ux_{t} + Ws_{t} \tag{20} \\
s_{t-1} &=& f(net_{t-1}) \tag{21} \\
\frac{\partial net_{t}}{\partial net_{t-1}} &=& \frac{\partial net_{t}}{\partial s_{t-1}} \frac{\partial s_{t-1}}{\partial net_{t-1}} \tag{22} 
\end{eqnarray*}
$$
我们用$a$表示列向量，用$a^{T}$表示行向量。上式的第一项是向量函数对向量求导，其结果为Jacobian矩阵：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405231306.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405232938.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190405233026.png)

**循环层**将**误差项**反向传递到上一层网络，与普通的**全连接层**是完全一样的。在此仅简要描述一下。**循环层**的**加权输入**$net^{l}$与上一层的**加权输入**$net^{l-1}$关系如下：
$$
\begin{eqnarray*}
net^{l}_{t} &=& Ua^{t-1}_{t}+Ws_{t-1} \tag{37} \\
a^{l-1}_{t} &=& f^{l-1}(net^{l-1}_{t}) \tag{38} 
\end{eqnarray*}
$$
![](https://raw.githubusercontent.com/bovane/md_images/master/20190406213141.png)

### 权重梯度的计算

现在，我们终于来到了BPTT算法的最后一步：计算每个权重的梯度。

首先，我们计算误差函数$E$对权重矩阵$W$的梯度$\frac{\partial E}{\partial W}$

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406214036.png)

上图展示了我们到目前为止，在前两步中已经计算得到的量，包括每个时刻t **循环层**的输出值$s_{t}$,以及误差项$\delta_{t}$,只要知道了任意一个时刻的**误差项**$\delta_{t}$，以及上一个时刻循环层的输出值$s_{t-1}$,就可以按照下面的公式求出权重矩阵在t时刻的梯度：

![1554558929789](C:\Users\bovan\AppData\Roaming\Typora\typora-user-images\1554558929789.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406220056.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406220255.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406220356.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406220436.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406221425.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406221551.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190406221730.png)

### RNN梯度消失和梯度爆炸

不幸的是，实践中前面介绍的几种RNNs并不能很好的处理较长的序列。一个主要的原因是，RNN在训练中很容易发生**梯度爆炸**和**梯度消失**，这导致训练时梯度不能在较长序列中一直传递下去，从而使RNN无法捕捉到长距离的影响。

为什么RNN会产生梯度爆炸和消失问题呢？我们接下来将详细分析一下原因。
$$
\begin{aligned} \delta_{k}^{T} &=\delta_{t}^{T} \prod_{i=k}^{t-1} W \operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{i}\right)\right] \\\left\|\delta_{k}^{T}\right\| & \leqslant\left\|\delta_{t}^{T}\right\| \prod_{i=k}^{t-1}\|W\|\left\|\operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{i}\right)\right]\right\| \\ & \leqslant\left\|\delta_{t}^{T}\right\|\left(\beta_{W} \beta_{f}\right)^{t-k} \end{aligned}
$$
上式的定义为矩阵的模的上界。因为上式是一个指数函数，==如果t-k很大的话（也就是向前看很远的时候）==，会导致对应的**误差项**的值增长或缩小的非常快，这样就会导致相应的**梯度爆炸**和**梯度消失**问题（取决于大于1还是小于1）。

通常来说，**梯度爆炸**更容易处理一些。因为梯度爆炸的时候，我们的程序会收到NaN错误。我们也可以设置一个梯度阈值，当梯度超过这个阈值的时候可以直接截取。

**梯度消失**更难检测，而且也更难处理一些。总的来说，我们有三种方法应对梯度消失问题：

- Batch Normalization训练层
- 合理的初始化和Relu激活函数
- 选用其他RNN架构，如LSTM

### RNN 例子

## RNN应用

RNNs已经被在实践中证明对NLP是非常成功的，如词向量表达、语句合法性检查、词性标注等。后来RNNs的应用领域扩展到其他方面，比如语音识别、图像描述生成。

### 语言模型与文本生成(Language Modeling and Generating Text)

### 机器翻译(Machine Translation)

### 语音识别(Speech Recognition)

### 图像描述生成 (Generating Image Descriptions)

## 代码

[tensorflow中的练手RNN](https://zhuanlan.zhihu.com/p/28196873)

## 参考文献

[参考链接1](https://blog.csdn.net/loveliuzz/article/details/79156025)

[参考链接2](https://zhuanlan.zhihu.com/p/32755043)

[四种架构](https://zhuanlan.zhihu.com/p/27485750)

[循环神经网络历史](https://www.jiqizhixin.com/articles/2018-01-05-5)

[深入浅出RNN](https://zybuluo.com/hanbingtao/note/541458)



## 

