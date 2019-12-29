[TOC]

# Inception 结构

Inception结构是在2014年在GooLeNet中首次提出，由于简单的堆积网络会存在以下问题：

- （1）参数太多，如果训练数据集有限，很容易产生过拟合；
- （2）网络越大、参数越多，计算复杂度越大，难以应用；
- （3）网络越深，容易出现梯度弥散问题（梯度越往后穿越容易消失），难以优化模型。

解决这些问题的方法当然就是在增加网络深度和宽度的同时减少参数，为了减少参数，自然就想到将全连接变成稀疏连接。但是在实现上，全连接变成稀疏连接后实际计算量并不会有质的提升，因为大部分硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量少，但是计算所消耗的时间却很难减少。**那么，有没有一种方法既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能呢？**==Inception网络结构，就是构造一种“基础神经元”结构，来搭建一个稀疏性、高计算性能的网络结构。==

## Inception V1

Inception V1在2014的[GoogLeNet](http://noahsnail.com/2017/07/21/2017-07-21-GoogleNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)中率先提出，作者的思想是通过设计一个稀疏网络结构，**但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。**下图最原始Inception的基本结构：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330173431.png)

Inception架构的**主要想法**是考虑怎样近似卷积神经网络的最优稀疏结构并用最容易获得的密集组件进行覆盖，因为我们提前假设转换不变性，这意味着Inception网络结构将以卷积构建块为基础，因此需要做的就是找到最优的局部构造并在空间上重复它。 

该结构将CNN中常用的卷积（1x1，3x3，5x5）、池化操作（3x3）堆叠在一起（卷积、池化后的尺寸相同，将通道相加），==一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。==
网络卷积层中的网络能够提取输入的每一个细节信息，同时5x5的滤波器也能够覆盖大部分接受层的的输入。还可以进行一个池化操作，以减少空间大小，降低过度拟合。在这些层之上，在每一个卷积层后都要做一个ReLU操作，以增加网络的非线性特征。
**然而这个Inception原始版本，所有的卷积核都在上一层的所有输出上来做，而那个5x5的卷积核所需的计算量就太大了，造成了特征图的厚度很大**，为了避免这种情况，在3x3前、5x5前、max pooling后分别加上了1x1的卷积核，以起到了降低特征图厚度的作用，这也就形成了Inception v1的网络结构，如下图所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330173514.png)

**1x1卷积的主要目的是为了减少维度，还用于修正线性激活（ReLU）**。GoogLeNet基于Inception V1构建的网络一共有22层，其网络结构如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330173232.png)

### 网络结构解读

- GoogLeNet采用了模块化的结构（Inception结构），方便增添和修改；
- 网络最后采用了average pooling（平均池化）来代替全连接层，该想法来自NIN（Network in Network），事实证明这样可以将准确率提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整。
- 虽然移除了全连接，但是网络中依然使用了Dropout ; 
- 为了避免梯度消失，网络额外增加了2个辅助的$softmax$用于向前传导梯度（辅助分类器）。辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的$softmax$会被去掉。
- 网络具体参数配置如下：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330174640.png)

### Inception V1总结

Inception 模块有用的一个方面在于它允许显著增加每个阶段的单元数量，却不会出现计算复杂度不受控制的情况(因为在尺寸较大的快进行卷积运算时会通过降维实现)。另一方面该结构遵循了实践直觉——视觉信息在不同尺度上处理然后再聚合，这样下一阶段可以从不同尺度同时抽象特征。

实验的良好表现证明了通过容易获得的密集构造块来近似期望的最优稀疏结果时改善计算机视觉神经网络的一种可行方法，尤其在计算资源有限的情况下。

## Inception V2

GoogLeNet设计的初衷就是要又准又快，而如果只是单纯的堆叠网络虽然可以提高准确率，但是会导致计算效率有明显的下降，==因此如何在不增加过多计算量的同时提高网络的表达能力就成为了一个问题。==Google团队**通过修改Inception的内部计算逻辑，提出了一种比较特殊的“卷积”计算结构**，由此产生Inception V2，相比于V1改进如下：

- 使用BN层，将每一层的输出都规范化到一个N(0,1)的正态分布，这将有助于训练，因为下一层不必学习输入数据中的偏移，并且可以专注与如何更好地组合特征（也因为在v2里有较好的效果，BN层几乎是成了深度网络的必备）
- 使用2个3x3的卷积代替5x5的卷积，这样既可以获得相同的感受野(经过2个3x3卷积得到的特征图大小等于1个5x5卷积得到的特征图)，还具有更少的参数，还间接增加了网络的深度
- 提出任意nxn的卷积都可以通过1xn卷积后接nx1卷积来替代，但是这需要在中等规模的特征图才有较好的效果(建议在12-20)之间

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330192514.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330192617.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330194618.png)

总体而言，Inception V2在V1的基础上进一步加深网络，以及采用先进的Batch Normalization方法。

## Inception V3

Inception V3最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处**，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，**还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块。同时V3还讨论了inception模块之间特征图的缩小的问题，主要有两种方式如下所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330195741.png)

左图先进行Pooling操作再进行Inception操作，这种方法参数量和计算量都较少，但是这样会导致表达瓶颈问题，**也就是说特征图的大小不应该出现急剧的衰减**(只经过一层就骤降)。==如果出现急剧缩减，将会丢失大量的信息，对模型的训练造成困难。== 右图先进行Inception操作，再进行Pooling操作，这样明显会导致参数量较多。因此为了既有良好的特征表示功能又让参数量较少，在Inception V3中提出一种Inception操作和Pooling操作并行化的方法。网络结构如下：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330200708.png)

Inception V3网络总结构图如下：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330201345.png)

[Inception Net-V3结构图](https://www.jianshu.com/p/3bbf0675cfce)

## Inception-ResNet

Inception V4研究了Inception模块与残差连接的结合。ResNet结构大大地加深了网络深度，还极大地提升了训练速度，同时性能也有提升.Inception V4主要利用残差连接（Residual Connection）来改进V3结构，得到Inception-ResNet-v1，Inception-ResNet-v2，Inception-v4网络。这里举一个例子即可，如下：

残差结构：![](https://raw.githubusercontent.com/bovane/md_images/master/20190330202807.png)

Inception+Resnet结构：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330202916.png)

最终网络结构，通过类似20个的Ince-Resnet组合而成。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190330203047.png)

参考文献：

[Inception结构分析](https://my.oschina.net/u/876354/blog/1637819)

[paper](http://noahsnail.com/2017/07/21/2017-07-21-GoogleNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

[Inception系列结构](https://zhuanlan.zhihu.com/p/30756181)

 