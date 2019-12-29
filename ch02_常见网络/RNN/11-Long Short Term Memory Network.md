[TOC]

# Long Short Term Memory Network

在实际应用中，原始RNN很难处理长距离的依赖。在本文中，我们将介绍一种改进之后的循环神经网络：**长短时记忆网络(Long Short Term Memory Network, LSTM)**，它成功的解决了原始循环神经网络的缺陷，成为当前最流行的RNN，在语音识别、图片描述、自然语言处理等许多领域中成功应用。

## LSTM产生的背景

LSTM提出是为了处理长距离的依赖问题，所谓长依赖就是我们需要的关键信息离上下文比较远，比如在下面的这个例子中：

==我== 昨天 上学 迟到 了 ，老师 批评 了 ____。

**关键信息 我 离 我们需要填空的地方比较远**，原始RNN在训练时会很容易出现梯度消失的情况，导致无法训练。在误差项沿时间反向传播的公式中：
$$
\delta_{k}^{T}=\delta_{t}^{T} \prod_{i=k}^{t-1} \operatorname{diag}\left[f^{\prime}\left(\mathbf{n e t}_{i}\right)\right] W \tag{1}
$$
我们可以根据下面的不等式，来获取$\delta_{t}^{T}$的模的上界（模可以看做对中每一项$\delta_{t}^{T}$值的大小的度量）：
$$
\begin{equation}
\begin{aligned}\left\|\delta_{k}^{T}\right\| & \leqslant\left\|\delta_{t}^{T}\right\| \prod_{i=k}^{t-1}\left\|\operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{i}\right)\right]\right\| W\| \\ & \leqslant\left\|\delta_{t}^{T}\right\|\left(\beta_{f} \beta_{W}\right)^{t-k} \end{aligned}
\end{equation}
$$


我们可以看到，误差项$\delta$从t时刻传递到k时刻，其值的上界是$\beta_{f}\beta_{W}$的指数函数。显然，除非乘积$\beta_{f}\beta_{W}$的值位于1附近，否则，当t-k很大时（也就是误差传递很多个时刻时），整个式子的值就会变得极小（当乘积$\beta_{f}\beta_{W}$小于1）或者极大（当乘积$\beta_{f}\beta_{W}$大于1），前者就是**梯度消失**，后者就是**梯度爆炸**。虽然科学家们搞出了很多技巧（比如怎样初始化权重），让$\beta_{f}\beta_{W}$的值尽可能贴近于1，终究还是难以抵挡指数函数的威力。**梯度消失**到底意味着什么？之前我们已证明，权重数组W最终的梯度是各个时刻的梯度之和，即：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413192742.png)

我们就可以看到，从上图的t-3时刻开始，梯度已经几乎减少到0了。**那么，从这个时刻开始再往之前走，得到的梯度（几乎为零）就不会对最终的梯度值有任何贡献，这就相当于无论t-3时刻之前的网络状态h是什么，在训练中都不会对权重数组W的更新产生影响，**也就是网络事实上已经忽略了t-3时刻之前的状态。这就是原始RNN无法处理长距离依赖的原因。

## LSTM核心思想

**长短时记忆网络**的核心思路比较简单。原始RNN的隐藏层只有一个状态，即h，既然它对于短期的输入非常敏感，无法记录长期的输入，**那么，假如我们再增加一个状态，即c，让它来保存长期的状态，**那么问题不就解决了么？实际上LSTM也是如此考虑的，引入门控的概念——记忆们和遗忘门，如下图所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413193358.png)

新增加的状态c，称为**单元状态(cell state)**。我们把上图按照时间维度展开：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413193432.png)

我们可以看出，==在t时刻，LSTM的输入有三个==：当前时刻网络的输入值$x_{t}$、上一时刻LSTM的输出值$h_{t-1}$、以及上一时刻的单元状态$c_{t-1}$；**LSTM的输出有两个：当前时刻LSTM输出值$h_{t}$、和当前时刻的单元状态$c_{t}$。**注意$x,h,c$都是**向量**。

LSTMs 最关键的地方在于 cell（整个绿色的框就是一个 cell） 的状态 和 结构图上面的那条横穿的水平线。

==cell 状态的传输就像一条传送带，向量从整个 cell 中穿过，只是做了少量的线性操作。==这种结构能够很轻松地实现信息从整个 cell 中穿过而不做改变。（注：这样我们就可以实现了长时期的记忆保留了）

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413195150.png)

若**只有上面的那条水平线是没办法实现添加或者删除信息的。**而是通过一种叫做 **门（gates）** 的结构来实现的。

**门** 可以实现选择性地让信息通过，主要是通过一个 sigmoid 的神经层 和一个逐点相乘的操作来实现的。sigmoid 层输出（是一个向量）的每个元素都是一个在 0 和 1 之间的实数，表示让对应信息通过的权重（或者占比）。比如， 0 表示“不让任何信息通过”， 1 表示“让所有信息通过”。**每个 LSTM 有三个这样的门结构，来实现保护和控制信息。**第一个开关，负责控制继续保存长期状态c；第二个开关，负责控制把即时状态输入到长期状态c；第三个开关，负责控制是否把长期状态c作为当前的LSTM的输出。三个开关的作用如下图所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413193815.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413194837.png)

### 遗忘门(forget gate layer)

==首先==是 LSTM 要决定让那些信息继续通过这个 cell，这是通过一个叫做“forget gate layer ”的sigmoid 神经层来实现的。它的输入是$ h_{t-1} 和 x_t $，**输出是一个数值都在 0，1 之间的向量**（向量长度和 cell 的状态 $ C_{t-1} $ 一样），表示让 $C_{t-1} $ 的各部分信息通过的比重。 0 表示“不让任何信息通过”， 1 表示“让所有信息通过”。

回到我们上面提到的语言模型中，我们要根据**所有**的上文信息来预测下一个词。这种情况下，每个 cell 的状态中都应该包含了当前主语的性别信息（保留信息），这样接下来我们才能够正确地使用代词。但是当我们又开始描述一个新的主语时，就应该把上文中的主语性别给忘了才对(忘记信息)。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413200625.png)

### 输入门

==下一步==是决定让多少新的信息$x$加入到 cell 状态 中来。实现这个需要包括两个 步骤：首先，一个叫做“input gate layer ”的 sigmoid 层决定哪些信息需要更新；一个 tanh 层生成一个向量，也就是备选的用来更新的内容，$\tilde{C}_{t}$ 。在下一步，我们把这两部分联合起来，对 cell 的状态进行一个更新。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413200932.png)

在我们的语言模型的例子中，**我们想把新的主语性别信息添加到 cell 状态中**，来替换掉老的状态信息。有了上述的结构，我们就能够更新 cell 状态了， 即把$ C_{t-1} $更新为 $C_{t} $。 从结构图中应该能一目了然， 首先我们把旧的状态 $C_{t-1} 和和和 f_t 相乘，把一些不想保留的信息忘掉。 i_t * \tilde{C_{t}} $。这部分信息就是我们要添加的新内容。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413201145.png)

### **输出门**

==最后，==我们需要来决定输出什么值了。这个输出主要是依赖于 cell 的状态$ C_t$，但是又不仅仅依赖于 $C_t $，而是需要经过一个过滤的处理。首先，我们还是使用一个sigmoid层来（计算出）决定，而是需要经过一个过滤的处理。首先，我们还是使用一个 sigmoid 层来（计算出）决定，而是需要经过一个过滤的处理。**首先**，我们还是使用一个sigmoid层来（计算出）决定$C_t$ 中的哪部分信息会被输出。**接着**，我们把中的哪部分信息会被输出。接着，我们把中的哪部分信息会被输出。接着，我们把 $C_t $通过一个 tanh 层（把数值都归到 -1 和 1 之间），然后把 tanh 层的输出和 sigmoid 层计算出来的权重相乘，这样就得到了最后输出的结果。

在语言模型例子中，假设我们的模型刚刚接触了一个代词，接下来可能要输出一个动词，这个输出可能就和代词的信息相关了。比如说，这个动词应该采用单数形式还是复数的形式，那么我们就得把刚学到的和代词相关的信息都加入到 cell 状态中来，才能够进行正确的预测。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190413201239.png)

## LSTM前向计算

前面描述的开关是怎样在算法中实现的呢？这就用到了**门（gate）**的概念。门实际上就是一层**全连接层**，它的输入是一个向量，输出是一个0到1之间的实数向量。假设**W**是门的权重向量，**b**是偏置项，那么门可以表示为：
$$
\mathbf{f}_{t}=\sigma\left(W_{f} \cdot\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{f}\right) \tag{式1}
$$
==门的使用，就是用门的输出向量按元素乘以我们需要控制的那个向量。==因为门的输出是0到1之间的实数向量，那么，当门输出为0时，任何向量与之相乘都会得到0向量，这就相当阻塞掉所有信息；输出为1时，任何向量与之相乘都不会有任何改变，这相当于所有的信息都能通过。**由于sigmoid函数的值域是(0,1)，所以门的状态都是半开半闭的。**

LSTM用两个门来控制单元状态c的内容，一个是**遗忘门（forget gate）**，它决定了上一时刻的单元状态$c_{t-1}$有多少保留到当前时刻$c_{t}$；另一个是**输入门（input gate）**，它决定了当前时刻网络的输入$x_{t}$有多少保存到单元状态$c_{t}$。LSTM用**输出门（output gate）**来控制单元状态$c_{t}$有多少输出到LSTM的当前输出值$h_{t}$。

**<font color=red>我们先来看看遗忘门</font>**
$$
\mathbf{f}_{t}=\sigma\left(W_{f} \cdot\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{f}\right) \tag{式1}
$$
其中$W_{f},b_{f}$分别为遗忘门的权重和偏置，$\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]$表示把两个向量连接成一个更长的向量。事实上，权重矩阵$W_{f}$都是两个矩阵拼接而成的：一个是$W_{fh}$，它对应着输入项$h_{t-1}$，其维度为$d_{c} \times d_{x}$；一个是$W_{fx}$，它对应着输入项$x_{t}$，其维度为$d_{c} \times d_{x}$,$W_{f}$可以写为：
$$
\begin{aligned}\left[W_{f}\right] \left[ \begin{array}{c}{\mathbf{h}_{t-1}} \\ {\mathbf{x}_{t}}\end{array}\right] &=\left[ \begin{array}{cc}{W_{f h}} & {W_{f x}}\end{array}\right] \left[ \begin{array}{c}{\mathbf{h}_{t-1}} \\ {\mathbf{x}_{t}}\end{array}\right] \\ &=W_{f h} \mathbf{h}_{t-1}+W_{f x} \mathbf{x}_{t} \end{aligned}
$$
==注意向量拼接符号和矩阵拼接符号。== 

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415182741.png)

**<font color=red>接着看看输入门</font>**
$$
\mathbf{i}_{t}=\sigma\left(W_{i} \cdot\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{i}\right) \tag{式2}
$$
其中$W_{i},b_{i}$分别为输入门的权重矩阵和偏置。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415183339.png)

我们计算==用于描述当前输入的单元状态==$\tilde{\mathbf{c}}_{t}$它是根据上一次的输出和本次输入来计算的
$$
\tilde{\mathbf{c}}_{t}=\tanh \left(W_{c} \cdot\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{c}\right) \tag{式3}
$$
下面是$\tilde{\mathbf{c}}_{t}$ 的计算图

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415183949.png)

现在，我们计算当前时刻的单元状态$c_{t}$。它是由**上一次的单元状态**$c_{t-1}$按元素乘以遗忘门$f_{t}$，==再用当前输入的单元状态==$\tilde{\mathbf{c}}_{t}$按元素乘以输入门，再将两个积加和产生的：
$$
\mathbf{c}_{t}=f_{t} \circ \mathbf{c}_{t-1}+i_{t} \circ \tilde{\mathbf{c}}_{t} \tag{式4}
$$
符号$o$表示**按元素乘**，所谓按元素乘如下所示：
$$
\mathbf{a} \circ \mathbf{b}=\left[ \begin{array}{c}{a_{1}} \\ {a_{2}} \\ {a_{3}} \\ {\dots} \\ {a_{n}}\end{array}\right] \circ \left[ \begin{array}{c}{b_{1}} \\ {b_{2}} \\ {b_{3}} \\ {\dots} \\ {b_{n}}\end{array}\right]=\left[ \begin{array}{c}{a_{1} b_{1}} \\ {a_{2} b_{2}} \\ {a_{3} b_{3}} \\ {\ldots} \\ {a_{n} b_{n}}\end{array}\right]
$$
当$o$作用于两个**矩阵**时，**两个矩阵对应位置的元素相乘。按元素乘可以在某些情况下简化矩阵和向量运算。**例如，当一个对角矩阵右乘一个矩阵时，相当于用对角矩阵的对角线组成的向量按元素乘那个矩阵：
$$
\operatorname{diag}[\mathbf{a}] X=\mathbf{a} \circ X
$$
当一个行向量右乘一个对角矩阵时，相当于这个行向量按元素乘那个矩阵对角线组成的向量：
$$
\mathbf{a}^{T} \operatorname{diag}[\mathbf{b}]=\mathbf{a} \circ \mathbf{b}
$$
<font color=red>下面是$c_{t}$的计算。</font>

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415184420.png)

这样，我们就把LSTM关于当前的记忆$\tilde{\mathbf{c}}_{t}$和长期的记忆$c_{t-1}$组合在一起，形成了新的单元状态$c_{t}$。**由于遗忘门的控制，它可以保存很久很久之前的信息，**==由于输入门的控制，它又可以避免当前无关紧要的内容进入记忆。==下面，我们要看看<font color=red>输出门，它控制了长期记忆对当前输出的影响</font>。
$$
\mathbf{o}_{t}=\sigma\left(W_{o} \cdot\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{o}\right) \tag{式5}
$$
下图表示输出门的计算：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415185416.png)

LSTM最终的输出，是由输出门和单元状态cell共同确定的：
$$
\mathbf{h}_{t}=\mathbf{o}_{t} \circ \tanh \left(\mathbf{c}_{t}\right) \tag{式6}
$$
==下图表示LSTM最终输出的计算：==

![](https://raw.githubusercontent.com/bovane/md_images/master/20190415185751.png)

这样我们就完成LSTM的前向计算过程。

## LSTM的训练

LSTM的训练算法仍然是反向传播算法，对于这个算法，我们已经非常熟悉了。主要有下面三个步骤：

- 前向计算每个神经元的输出值，对于LSTM来说，即$f_{t},i_{t},c_{t},o_{t},h_{t}$五个向量的值。

- 反向计算每个神经元的**误差项** $\delta$值。与**循环神经网络**一样，LSTM误差项的反向传播也是包括两个方向：一个是沿时间的反向传播，即从当前t时刻开始，计算每个时刻的误差项；一个是将误差项向上一层传播。

- 根据相应的误差项，计算每个权重的梯度。

首先，我们对推导中用到的一些公式、符号做一下必要的说明。

接下来的推导中，我们**设定gate的激活函数为sigmoid函数**，==输出的激活函数为tanh函数==。他们的导数分别为：
$$
\begin{aligned} \sigma(z) &=y=\frac{1}{1+e^{-z}} \\ \sigma^{\prime}(z) &=y(1-y) \\ \tanh (z) &=y=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \\ \tanh ^{\prime}(z) &=1-y^{2} \end{aligned}
$$
LSTM需要学习的参数共有8组，分别是：遗忘门的权重矩阵 $W_{f}$ 和偏置项 $b_{f}$、输入门的权重矩阵$W_{i}$和偏置项$b_{i}$、输出门的权重矩阵$W_{o}$和偏置项$b_{o}$，以及计算单元状态的权重矩阵$W_{c}$和偏置项$b_{c}$。

我们假设在t时刻，LSTM的输出值为$h_{t}$。我们定义t时刻的误差项$\delta_{t}$为:
$$
\delta_{t} \stackrel{d e f}{=} \frac{\partial E}{\partial \mathbf{h}_{t}}
$$
注意，我们这里假设误差项是损失函数对输出值的导数，而不是对加权输入$net_{t}^{l}$的导数。因为LSTM有四个加权输入，分别对应$f_{t},i_{t},c_{t},o_{t}$，我们==希望往上一层传递一个误差项而不是四个==。但是我们仍然需要定义出这四个加权输入，以及他们对应的误差项。
$$
\begin{aligned} \text { net }_{f, t} &=W_{f}\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{f} \\ &=W_{f h} \mathbf{h}_{t-1}+W_{f x} \mathbf{x}_{t}+\mathbf{b}_{f} \end{aligned}
$$

$$
\begin{aligned} \mathbf{n e t}_{i, t} &=W_{i}\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{i} \\ &=W_{i h} \mathbf{h}_{t-1}+W_{i x} \mathbf{x}_{t}+\mathbf{b}_{i} \end{aligned}
$$

$$
\begin{aligned} \mathbf{n e t}_{\tilde{c}, t} &=W_{c}\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{c} \\ &=W_{c h} \mathbf{h}_{t-1}+W_{c x} \mathbf{x}_{t}+\mathbf{b}_{c} \end{aligned}
$$

$$
\begin{aligned} \mathbf{n e t}_{o, t} &=W_{o}\left[\mathbf{h}_{t-1}, \mathbf{x}_{t}\right]+\mathbf{b}_{o} \\ &=W_{o h} \mathbf{h}_{t-1}+W_{o x} \mathbf{x}_{t}+\mathbf{b}_{o} \end{aligned}
$$

$$
\delta_{f, t} \stackrel{d e f}{=} \frac{\partial E}{\partial \mathbf{n e t}_{f, t}}
$$

$$
\delta_{i, t} \stackrel{d e f}{=} \frac{\partial E}{\partial \mathbf{n e t}_{i, t}}
$$

$$
\delta_{\tilde{c}, t} \stackrel{d e f}{=} \frac{\partial E}{\partial \mathbf{n e t}_{\tilde{c}, t}}
$$

$$
\delta_{o, t} \stackrel{d e f}{=} \frac{\partial E}{\partial \mathbf{n e t}_{o, t}}
$$

**<font color=red>接下来我们看看误差项时间的反向传递</font>**

我们知道沿着时间的误差项传播，就是要计算出$t-1$时刻的误差项$\delta_{t-1}$，以及$t-2,t-3$时刻，一个时间步一个时间步的向前传递。
$$
\begin{aligned} \delta_{t-1}^{T} &=\frac{\partial E}{\partial \mathbf{h}_{\mathbf{t}-1}} \\ &=\frac{\partial E}{\partial \mathbf{h}_{\mathbf{t}}} \frac{\partial \mathbf{h}_{\mathbf{t}}}{\partial \mathbf{h}_{\mathbf{t}-\mathbf{1}}} \\ &=\delta_{t}^{T} \frac{\partial \mathbf{h}_{\mathbf{t}}}{\partial \mathbf{h}_{\mathbf{t}-\mathbf{1}}} \end{aligned}
$$
我们知道，$\frac{\partial \mathbf{h}_{t}}{\partial \mathbf{h}_{t-1}}$是一个是一个[Jacobian矩阵](https://zh.wikipedia.org/zh-hans/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)。如果隐藏层h的维度是N的话，那么它就是$N \times N$一个矩阵。为了求出它，我们需要知道$h_{t}$,如下：
$$
\begin{aligned} \mathbf{h}_{t} &=\mathbf{o}_{t} \circ \tanh \left(\mathbf{c}_{t}\right) \\ \mathbf{c}_{t} &=\mathbf{f}_{t} \circ \mathbf{c}_{t-1}+\mathbf{i}_{t} \circ \tilde{\mathbf{c}}_{t} \end{aligned}
$$
显然，显然$f_{t},i_{t},c_{t},o_{t}，$$\tilde{\mathbf{c}}_{t}$都是的函数，那么，利用全导数公式可得：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190424221147.png)

下面，我们要把**式7**中的每个偏导数都求出来。根据**式6**，我们可以求出：
$$
\frac{\partial \mathbf{h}_{\mathbf{t}}}{\partial \mathbf{o}_{t}}=\operatorname{diag}\left[\tanh \left(\mathbf{c}_{t}\right)\right]
$$

$$
\frac{\partial \mathbf{h}_{\mathbf{t}}}{\partial \mathbf{c}_{t}}=\operatorname{diag}\left[\mathbf{o}_{t} \circ\left(1-\tanh \left(\mathbf{c}_{t}\right)^{2}\right)\right]
$$

根据上面的计算公式，我们可以求出：
$$
\begin{aligned} \frac{\partial \mathbf{c}_{t}}{\partial \mathbf{f}_{\mathrm{t}}} &=\operatorname{diag}\left[\mathbf{c}_{t-1}\right] \\ \frac{\partial \mathbf{c}_{t}}{\partial \mathbf{i}_{\mathbf{t}}} &=\operatorname{diag}\left[\tilde{\mathbf{c}}_{t}\right] \\ \frac{\partial \mathbf{c}_{t}}{\partial \tilde{\mathbf{c}}_{\mathrm{t}}} &=\operatorname{diag}\left[\mathbf{i}_{t}\right] \end{aligned}
$$
由于下面的式子：
$$
\begin{aligned} \mathbf{o}_{t} &=\sigma\left(\mathbf{n e t}_{o, t}\right) \\ \mathbf{n e t}_{o, t} &=W_{o h} \mathbf{h}_{t-1}+W_{o x} \mathbf{x}_{t}+\mathbf{b}_{o} \end{aligned}
$$

$$
\begin{aligned} \mathbf{f}_{t} &=\sigma\left(\mathbf{n e t}_{f, t}\right) \\ \mathbf{n e t}_{f, t} &=W_{f h} \mathbf{h}_{t-1}+W_{f x} \mathbf{x}_{t}+\mathbf{b}_{f} \end{aligned}
$$

$$
\begin{aligned} \mathbf{i}_{t} &=\sigma\left(\mathbf{n} \mathbf{e} \mathbf{t}_{i, t}\right) \\ \mathbf{n e t}_{i, t} &=W_{i h} \mathbf{h}_{t-1}+W_{i x} \mathbf{x}_{t}+\mathbf{b}_{i} \end{aligned}
$$

$$
\begin{aligned} \tilde{\mathbf{c}}_{t} &=\tanh \left(\mathbf{n} \mathbf{e} \mathbf{t}_{\tilde{c}, t}\right) \\ \mathbf{n e t}_{\tilde{c}, t} &=W_{c h} \mathbf{h}_{t-1}+W_{c x} \mathbf{x}_{t}+\mathbf{b}_{c} \end{aligned}
$$

我们很容易得出：
$$
\begin{aligned} \frac{\partial \mathbf{o}_{t}}{\partial \mathbf{n e t}_{o, t}} &=\operatorname{diag}\left[\mathbf{o}_{t} \circ\left(1-\mathbf{o}_{t}\right)\right] \\ \frac{\partial \mathbf{n e t}_{o, t}}{\partial \mathbf{h}_{\mathbf{t}-1}} &=W_{o h} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial \mathbf{f}_{t}}{\partial \mathbf{n} \mathbf{e} t}_{f, t} &=\operatorname{diag}\left[\mathbf{f}_{t} \circ\left(1-\mathbf{f}_{t}\right)\right] \\ \frac{\partial \mathbf{n e t}_{f, t}}{\partial \mathbf{h}_{t-1}} &=W_{f h} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial \mathbf{i}_{t}}{\partial \mathbf{n} \mathbf{e} t_{i, t}} &=\operatorname{diag}\left[\mathbf{i}_{t} \circ\left(1-\mathbf{i}_{t}\right)\right] \\ \frac{\partial \mathbf{n e t}_{i, t}}{\partial \mathbf{h}_{t-1}} &=W_{i h} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial \tilde{\mathbf{c}}_{t}}{\partial \mathbf{n e t}_{\tilde{c}, t}} &=\operatorname{diag}\left[1-\tilde{\mathbf{c}}_{t}^{2}\right] \\ \frac{\partial \mathbf{n e t}_{\tilde{c}, t}}{\partial \mathbf{h}_{t-1}} &=W_{c h} \end{aligned}
$$

![](https://raw.githubusercontent.com/bovane/md_images/master/20190427151004.png)

**式8**到**式12**就是将误差沿时间反向传播一个时刻的公式。有了它，我们可以写出将误差项向前传递到任意k时刻的公式如下：
$$
\delta_{k}^{T}=\prod_{j=k}^{t-1} \delta_{o, j}^{T} W_{o h}+\delta_{f, j}^{T} W_{f h}+\delta_{i, j}^{T} W_{i h}+\delta_{\tilde{c}, j}^{T} W_{c h}
$$

### 将误差项传递到上一层

我们假设当前为第$l$层，定义$l-1$层的误差项是误差函数对$l-1$层**加权输入**的导数，即：
$$
\delta_{t}^{l-1} \stackrel{d e f}{=} \frac{\partial E}{\operatorname{net}_{t}^{l-1}}
$$
本次LSTM的输入$x_{t}$由下面的公式计算：
$$
\mathbf{x}_{t}^{l}=f^{l-1}\left(\mathbf{n} \mathbf{e} \mathbf{t}_{t}^{l-1}\right)
$$
其中$f^{l-1}$为第$l-1$层的激活函数

![](https://raw.githubusercontent.com/bovane/md_images/master/20190427151340.png)

**式14**就是将误差传递到上一层的公式。

### 权重梯度的计算

对于$W_{fh},W_{ih},W_{oh},W_{ch}$的权重梯度，我们知道它的梯度是各个时刻梯度之和。因此我们首先求出它们在t时刻的梯度，然后再求出他们最终的梯度。由于我们已经求得了误差项$\delta_{o,t},\delta_{f,t},\delta_{i,t},\delta_{c`,t}$，很容易求出t时刻$W_{fh},W_{ih},W_{oh},W_{ch}$
$$
\begin{aligned} \frac{\partial E}{\partial W_{o h, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{o, t}} \frac{\partial \mathbf{n e t}_{o, t}}{\partial W_{o h, t}} \\ &=\delta_{o, t} \mathbf{h}_{t-1}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{f h, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{f, t}} \frac{\partial \mathbf{n e t}_{f, t}}{\partial W_{f h, t}} \\ &=\delta_{f, t} \mathbf{h}_{t-1}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{i h, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{i, t}} \frac{\partial \mathbf{n e t}_{i, t}}{\partial W_{i h, t}} \\ &=\delta_{i, t} \mathbf{h}_{t-1}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{c h, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{\overline{c}, t}} \frac{\partial \mathbf{n e t}_{\overline{c}, t}}{\partial W_{c h, t}} \\ &=\delta_{\overline{c}, t} \mathbf{h}_{t-1}^{T} \end{aligned}
$$

将各个时刻的梯度加在一起，就能得到最终的梯度：
$$
\begin{aligned} \frac{\partial E}{\partial W_{o h}} &=\sum_{j=1}^{t} \delta_{o, j} \mathbf{h}_{j-1}^{T} \\ \frac{\partial E}{\partial W_{f h}} &=\sum_{j=1}^{t} \delta_{f, j} \mathbf{h}_{j-1}^{T} \\ \frac{\partial E}{\partial W_{i h}} &=\sum_{j=1}^{t} \delta_{i, j} \mathbf{h}_{j-1}^{T} \\ \frac{\partial E}{\partial W_{c h}} &=\sum_{j=1}^{t} \delta_{\tilde{c}, \mathbf{h}} \mathbf{h}_{j-1}^{T} \end{aligned}
$$
对于偏置$b_{f},b_{i},b_{c},b_{o}$的梯度，也是将各个时刻的梯度加在一起。下面是各个时刻的偏置项梯度：
$$
\begin{aligned} \frac{\partial E}{\partial \mathbf{b}_{o, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{o, t}} \frac{\partial \mathbf{n e t}_{o, t}}{\partial \mathbf{b}_{o, t}} \\ &=\delta_{o, t} \\ \frac{\partial E}{\partial \mathbf{b}_{f, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{f, t}} \frac{\partial \mathbf{n e t}_{f, t}}{\partial \mathbf{b}_{f, t}} \\ &=\delta_{f, t} \\ \frac{\partial E}{\partial \mathbf{b}_{i, t}} &=\frac{\partial E}{\partial \mathbf{n e t}_{i, t}} \frac{\partial \mathbf{n e t}_{i, t}}{\partial \mathbf{b}_{i, t}} \\ &=\delta_{i, t} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial \mathbf{b}_{c, t}} &=\frac{\partial E}{\partial \mathbf{n} e \mathbf{t}_{\tilde{c}, t}} \frac{\partial \mathbf{n e t}_{\tilde{c}, t}}{\partial \mathbf{b}_{c, t}} \\ &=\delta_{\tilde{c}, t} \end{aligned}
$$

下面是最终的偏置项梯度，即将各个时刻的偏置项梯度加在一起：
$$
\begin{aligned} \frac{\partial E}{\partial \mathbf{b}_{o}} &=\sum_{j=1}^{t} \delta_{o, j} \\ \frac{\partial E}{\partial \mathbf{b}_{i}} &=\sum_{j=1}^{t} \delta_{i, j} \\ \frac{\partial E}{\partial \mathbf{b}_{f}} &=\sum_{j=1}^{t} \delta_{f, j} \\ \frac{\partial E}{\partial \mathbf{b}_{c}} &=\sum_{j=1}^{t} \delta_{\tilde{c}, j} \end{aligned}
$$
对于$W_{fx},W_{ix},W_{cx},W_{ox}$的权重梯度，只需要根据相应的误差项直接计算即可：
$$
\begin{aligned} \frac{\partial E}{\partial W_{o x}} &=\frac{\partial E}{\partial \mathbf{n e t}_{o, t}} \frac{\partial \mathbf{n e t}_{o, t}}{\partial W_{o x}} \\ &=\delta_{o, t} \mathbf{x}_{t}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{f x}} &=\frac{\partial E}{\partial \mathbf{n} e \mathbf{t}_{f, t}} \frac{\partial \mathbf{n e t}_{f, t}}{\partial W_{f x}} \\ &=\delta_{f, t} \mathbf{x}_{t}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{i x}} &=\frac{\partial E}{\partial \mathbf{n e t}_{i, t}} \frac{\partial \mathbf{n e t}_{i, t}}{\partial W_{i x}} \\ &=\delta_{i, t} \mathbf{x}_{t}^{T} \end{aligned}
$$

$$
\begin{aligned} \frac{\partial E}{\partial W_{c x}} &=\frac{\partial E}{\partial \mathbf{n e t}_{\tilde{c}, t}} \frac{\partial \mathbf{n e t}_{\tilde{c}, t}}{\partial W_{c x}} \\ &=\delta_{\tilde{c}, t} \mathbf{x}_{t}^{T} \end{aligned}
$$

以上就是LSTM的训练算法的全部公式。其实这里面存在很多的重复模式，我们只需要理解一种就可以触类旁通到其它参数的训练。

## LSTM 实现

[STM实战之机场客流量预测](https://imyong.top/2018/06/13/LSTM%E5%AE%9E%E6%88%98%E4%B9%8B%E6%9C%BA%E5%9C%BA%E5%AE%A2%E6%B5%81%E9%87%8F%E9%A2%84%E6%B5%8B/)

[LSTM example-tensorflow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)

[Human Activity Recognition](https://github.com/healthDataScience/deep-learning-HAR)



## SUMMARY

### Relted knowledge

- 

## 参考文献

[详解LSTM](https://zybuluo.com/hanbingtao/note/581764)

[深入浅出LSTM](https://blog.csdn.net/Jerr__y/article/details/58598296)

[动态演示LSTM](https://www.atyun.com/30234.html)

