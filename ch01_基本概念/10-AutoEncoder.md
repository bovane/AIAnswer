[TOC]

# AutoEncoder

AutoEncoder是被称为自编码器的神经网络，它以一种无监督的方式学习数据的编码。一般而言自编码器的目的是通过训练网络来学习一组数据的表示（编码），该方法通常用于降低维数。

## 为什么我们需要AutoEncoder?

假设你参与了一个图像处理的项目，而目标就是设计一种算法来分析面部的表情推断情绪。算法的输入是一个 $256 \times 256$ 的灰度图，而输出则是一个情绪。例如，如果你输入下面的图片，期望出现一个标识为 happy 的算法结果。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190503141858.png)

如果我们直接将整个$256 \times 256$的图像输入，也就是我们的图像空间达到$6W$多维，这对计算机来说不是一件简单的事情。不仅大规模的输入非常难存储，移动和计算，而且他们会导致一些相当棘手的可有效计算性上的问题。其中比较显著的问题为维度爆炸和稀疏性。

### 维度爆炸(指数级增长)

让我们粗略看看在维度增长时，一个机器学习问题的难度如何增加的。根据 1982 年 C. J. Stone 的[这个](https://link.jianshu.com?t=http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf) 研究，用来训练模型的时间（非参数回归）最优情况下和 $m^{-p/(3p+d)}$ 成比例，其中 `m` 是数据点的数目，`d` 是数据的维度，`p` 则是依赖于使用的模型的参数（假设回归函数是 p 次可微的）简而言之，这个关系说明我们的数据的维度增加时需要指数级数量的训练样本。

我们可以从一个 Gutierrez 和 Osuna 提供的简单的可视化例子中看到这点。我们的学习算法将特征空间均匀地划分成 bin 并将训练样本画出来。接着我们赋给每个 bin 一个 label，基于在那个 bin 中占统治地位的样本的 label。最终，对每个新来的需要进行分类的样本，我们仅仅需要弄清楚在那个样本落入的 bin，以那个 bin 的 label 作为预测的结果！
 在这个例子中，我们刚开始挑选了单一的特征（一维的输入）并划分空间成 3 个简单的 bin：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190503142752.png)

从图中可以看出，在一维的时候，数据点是非常稠密的。而随着维数的增加（如2维，3维），整个数据点的空间就形成非常庞大的空间（由 3 个，变成 9 个，再变成 27 个），相应地，数据分布就变得相当稀疏了。

这个维数爆炸的问题并不是通过更加高效的算法或者性能提升的硬件就可以解决的！对许多机器学习任务，收集训练样本就是最耗费时间的部分，所以这个数学结果逼着我们精心确定分析的维度。如果我们能够将输入限制在一个相当小得维度时，我们可能将一个 不可行的（unfeasible）问题 变成 可行的（feasible）！

### 降维方法

通过无监督算法收集信息量最大的维度，降低稀疏性。AutoEncoder便是便是由 Geoffery Hinton 提出的一种实现无监督的神经网络。相比于传统的PCA降维方法，AutoEncoder有更好的效果。

## 什么是AutoEncoder?

**自动编码器(AutoEncoder)最开始作为一种数据的压缩方法**，其特点有: 

1)跟数据相关程度很高，这意味着自动编码器只能压缩与训练数据相似的数据，这个其实比较显然，因为使用神经网络提取的特征一般是高度相关于原始的训练集，使用人脸训练出来的自动编码器在压缩自然界动物的图片是表现就会比较差，因为它只学习到了人脸的特征，而没有能够学习到自然界图片的特征；

2)压缩后数据是有损的，这是因为在降维的过程中不可避免的要丢失掉信息；

到了2012年，人们发现在卷积网络中使用自动编码器做逐层预训练可以训练更加深层的网络，但是很快人们发现良好的初始化策略要比费劲的逐层预训练有效地多，2014年出现的Batch Normalization技术也是的更深的网络能够被被有效训练，到了15年底，通过残差(ResNet)我们基本可以训练任意深度的神经网络。

==现在自动编码器主要应用有两个方面，第一是数据去噪，第二是进行可视化降维。AutoEncoder也可以用于生成数据。==

## AutoEncoder的结构？

AutoEncoder包含两个过程，分别为Encoder过程（学习将来自输入层的数据压缩成code）和Decoder过程（将code解压缩成与原始数据紧密匹配的code）。如果我们学习的目的是如何去除噪声，那么AutoEncoder便用于数据去噪。另外如果我们的学习目的是用于生成数据，那么AutoEncoder便能用于数据增强。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190503144757.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190503151647.png)

输入的数据经过神经网络降维到一个编码(code)，接着又通过另外一个神经网络去解码得到一个与输入原数据一模一样的生成数据，然后通过去比较这两个数据，最小化他们之间的差异来训练这个网络中编码器和解码器的参数。当这个过程训练完之后，我们可以拿出这个解码器，随机传入一个编码(code)，希望通过解码器能够生成一个和原数据差不多的数据。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190503151839.png)

encoder 网络在训练和部署时候使用，而 decoder 网络只在训练的时候用。encoder 网络的作用是用来发现给定数据的压缩表示。

### 训练过程过程

$$
\begin{array}{l}{\phi : \mathcal{X} \rightarrow \mathcal{F}} \\ {\psi : \mathcal{F} \rightarrow \mathcal{X}} \\ {\phi, \psi=\underset{\phi, \psi}{\arg \min }\|X-(\psi \circ \phi) X\|^{2}}\end{array}
$$

Encoder时，我们将输入的数据压缩成一个code，用$\phi$表示encoder要学习的模型，Decoder时，我们将压缩成的code解码使之尽可能接近于原始输入X，用$\psi$表示decoder要学习的模型。

我们考虑只有一个隐藏层时的最简单情况：

- the encoder stage of an autoencoder takes the input$\mathbf{x} \in \mathbb{R}^{d}=\mathcal{X}$ and maps it to $\mathbf{z} \in \mathbb{R}^{p}=\mathcal{F} :$

$$
\mathbf{z}=\sigma(\mathbf{W} \mathbf{x}+\mathbf{b})
$$

其中$z$被称为code或者压缩表示

- the decoder stage of the autoencoder maps $\mathbf {z} $ to the *reconstruction* $\mathbf {x'}  $of the same shape as $\mathbf {x} $

$$
\mathbf{x}^{\prime}=\sigma^{\prime}\left(\mathbf{W}^{\prime} \mathbf{z}+\mathbf{b}^{\prime}\right)
$$

其中$\mathbf {x'} $被成为解压缩

- $$
  \mathcal{L}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|^{2}=\left\|\mathbf{x}-\sigma^{\prime}\left(\mathbf{W}^{\prime}(\sigma(\mathbf{W} \mathbf{x}+\mathbf{b}))+\mathbf{b}^{\prime}\right)\right\|^{2}
  $$



![](https://raw.githubusercontent.com/bovane/md_images/master/20190503155804.png)

## 自编码器的变种

[自编码器变种](https://www.atyun.com/17888.html)

## 参考文献

[什么是自编码器？](https://blog.csdn.net/qq_19528953/article/details/81048636)

[AutoEncoder-wikipedia](https://en.wikipedia.org/wiki/Autoencoder)