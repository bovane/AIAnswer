[TOC]

# Generative Adversarial Networks

GAN是“生成对抗网络”（Generative Adversarial Networks）的简称，由2014年还在蒙特利尔读博士的Ian Goodfellow引入深度学习领域。2016年，GAN热潮席卷AI领域顶级会议，从ICLR到NIPS，大量高质量论文被发表和探讨。Yann LeCun曾评价GAN是“**20年来机器学习领域最酷的想法**”。那么直观理解GAN到底是什么呢？

GAN是生成式模型的一种，所谓[生成式模型](https://www.zhihu.com/question/20446337)就是在预测一个物体时，你先要把它构造出来，最后求一个联合概率，从而判别新的样本。举个例子：我们希望区分一只羊到底是山羊还是绵羊，当利用生成模型时我们通过两步解决这个任务：

- 学习模型——根据山羊和绵羊的特征分别学习出一个模型
- 验证生成概率——从新羊中提取特征，分别放到提前生成好的山羊和绵羊模型，输出概率哪个大就是哪个

==现在比较流行的生成模型，其实可以分为三类：==

> 1. **生成对抗网络（GAN）**。这个是我们今天要重点介绍的内容。
>
> 2. **变分自动编码模型（VAE）**。它依靠的是传统的概率图模型的框架，通过一些适当的联合分布的概率逼近，简化整个学习过程，使得所学习到的模型能够很好地解释所观测到的数据。
>
> 3. **自回归模型（Auto-regressive）**。在这种模型里，我们简单地认为，每个变量只依赖于它的分布，只依赖于它在某种意义上的近邻。例如将自回归模型用在图像的生成上。那么像素的取值只依赖于它在空间上的某种近邻。现在比较流行的自回归模型，包括最近刚刚提出的像素CNN或者像素RNN，它们可以用于图像或者视频的生成。

**GAN 相比于其他生成式模型，有两大特点：** 

**1. 不依赖任何先验假设。**传统的许多方法会假设数据服从某一分布，然后使用极大似然去估计数据分布。 

**2. 生成 real-like 样本的方式非常简单。**GAN 生成 real-like 样本的方式通过生成器（Generator）的前向传播，而传统方法的采样方式非常复杂，有兴趣的同学可以参考下周志华老师的《机器学习》一书中对各种采样方式的介绍。 

## 1.GAN基本概念

### 1.1 GAN的思想

### 1.2 Generator & Discriminator

### 1.3 常见的目标函数及其比较

### 1.4 GAN 的评价指标

那么如何评价一个GAN Network是否是一个好的Network呢？目前常用的有以下几个指标：

## 2.GAN 训练

### 2.1 训练过程

GAN的训练包括训练Generator & Discriminator....

### 2.2 训练中存在的问题

### 2.3 稳定GAN训练的技巧

## 3.GAN的变种模型

### 3.1 条件GAN

### 3.2 CycleGAN

## 4.GAN应用场景

### 4.1 图像

### 4.2  序列生成

## 5.GAN 优缺点分析

## 6.未来研究方向

## 7.SUMMARY



## 参考文献

[GAN基本原理详解](https://blog.csdn.net/on2way/article/details/72773771)

[GAN万字长文综述-机器之心](https://www.jiqizhixin.com/articles/2019-03-19-12)

[GAN的应用、走向-雷锋网](https://www.leiphone.com/news/201701/Kq6FvnjgbKK8Lh8N.html)

[GAN在图像生成中的应用综述](http://www.twistedwg.com/2019/01/23/GAN_image_generation.html)

[GANs Paper](https://zhuanlan.zhihu.com/p/28504510)

[生成模型和判别模型理解](https://blog.csdn.net/zouxy09/article/details/8195017)

[GAN理解](https://blog.csdn.net/sxf1061926959/article/details/54630462)

