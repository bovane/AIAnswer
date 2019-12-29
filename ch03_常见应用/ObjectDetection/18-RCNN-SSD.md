[TOC]

# 基于深度学习的目标检测框架

## R-CNN系的两刀流

### R-CNN

[R-CNN](https://arxiv.org/abs/1311.2524)为基于卷积神经网络提取图片特征进行目标检测的开山之作，自那以后R-CNN系列网络占据了目标检测主流。R-CNN解决了传统检测方法的两个问题

**解决问题一、速度**
传统的区域选择使用滑窗，每滑一个窗口检测一次，相邻窗口信息重叠高，检测速度慢。R-CNN 使用一个启发式方法（Selective search），**先生成候选区域再检测，降低信息冗余程度**，从而提高检测速度。

**解决问题二、特征提取**
传统的手工提取特征鲁棒性差，限于如颜色、纹理等 `低层次`（Low level）的特征。而本文基于CNN的特征自动提取方法，可以提取不同层次的特征使得能够获得图像的更多信息。

==R-CNN的工作流程一共分为四步，如下所示：==

- 一张图像生成1K~2K个候选区域 
- 对每个候选区域，使用CNN提取特征 
- 特征送入每一类的SVM 分类器，判别是否属于该类 
- 使用回归器精细修正候选框位置 

![](https://raw.githubusercontent.com/bovane/md_images/master/20190513210418.png)

#### 候选区域生成

使用了Selective Search1方法从一张图像生成约2000-3000个候选区域。基本思路如下： 
- 使用一种过分割手段，将图像分割成小区域 
- 查看现有小区域，合并可能性最高的两个区域。重复直到整张图像合并成一个区域位置 
- 输出所有曾经存在过的区域，所谓候选区域

候选区域生成和后续步骤相对独立，实际可以使用任意算法进行。

在生成候选区域时，R-CNN采取一系列的合并规则

优先合并以下四种区域： 

- 颜色（颜色直方图）相近的 
- 纹理（梯度直方图）相近的 
- 合并后总面积小的 
- 合并后，总面积在其BBOX中所占比例大的

第三条，保证合并操作的尺度较为均匀，避免一个大区域陆续“吃掉”其他小区域。

```
例：设有区域a-b-c-d-e-f-g-h。较好的合并方式是：ab-cd-ef-gh -> abcd-efgh -> abcdefgh。 
不好的合并方法是：ab-c-d-e-f-g-h ->abcd-e-f-g-h ->abcdef-gh -> abcdefgh。
```

第四条，保证合并后形状规则。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190513211436.png)

为尽可能不遗漏候选区域，上述操作在多个颜色空间中同时进行（RGB,HSV,Lab等）。在一个颜色空间中，使用上述四条规则的不同组合进行合并。**所有颜色空间与所有规则的全部结果，在去除重复后，都作为候选区域输出。**

#### 类别判断
- 分类器 ：每一类目标，使用一个线性SVM二类分类器进行判别。输入为深度网络输出的4096维特征，输出是否属于此类。 由于负样本很多，使用hard negative mining方法。 
- 正样本 ：本类的真值标定框。 
- 负样本 ：考察每一个候选框，如果和本类所有标定框的重叠都小于0.3，认定其为负样本

#### 位置精修
目标检测问题的衡量标准是重叠面积：许多看似准确的检测结果，往往因为候选框不够准确，重叠面积很小。故需要一个位置精修步骤。 

- 回归器 ：对每一类目标，使用一个线性脊回归器进行精修。正则项λ=10000输入为深度网络pool5层的4096维特征，输出为xy方向的缩放和平移。 
- 训练样本 ：判定为本类的候选框中，和真值重叠面积大于0.6的候选框。

#### 实验结果以及缺点

该方法将 `PASCAL VOC` 上的检测率从 35.1% 提升到 53.7%，首次将深度学习引入目标检测领域。

**硬伤一、算力冗余**
先生成候选区域，再对区域进行卷积，这里有两个问题：其一是候选区域会有一定程度的重叠，对相同区域进行重复卷积；其二是每个区域进行新的卷积需要新的存储空间。

何恺明等人意识到这个可以优化，于是把**先生成候选区域再卷积**，变成了**先卷积后生成区域**。“简单地”改变顺序，不仅减少存储量而且加快了训练速度。

**硬伤二、图片缩放**

![](https://raw.githubusercontent.com/bovane/md_images/master/20190513214549.png)

**无论是剪裁（Crop）还是缩放（Warp），在很大程度上会丢失图片原有的信息导致训练效果不好**，如上图所示。直观的理解，把车剪裁成一个门，人看到这个门也不好判断整体是一辆车；把一座高塔缩放成一个胖胖的塔，人看到也没很大把握直接下结论。人都做不到，机器的难度就可想而知了。

### SPP NET

在[SPP NET](https://arxiv.org/abs/1406.4729)出现之前，所有的神经网络都是需要输入固定尺寸的图片，比如$224\times224（ImageNet）、32\times32(LenNet)、96\times96$等。==这样对于我们希望检测各种大小的图片的时候，需要经过crop，或者warp等一系列操作，这都在一定程度上导致图片信息的丢失和变形，限制了识别精确度。==而且，**从生理学角度出发，人眼看到一个图片时，大脑会首先认为这是一个整体，而不会进行crop和warp，**所以更有可能的是，我们的大脑通过搜集一些浅层的信息，在更深层才识别出这些任意形状的目标。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514205122.png)

**卷积层的参数和输入大小无关**，它仅仅是一个卷积核在图像上滑动，不管输入图像多大都没关系，只是对不同大小的图片卷积出不同大小的特征图，但是==全连接层的参数就和输入图像大小有关，因为它要把输入的所有像素点连接起来,需要指定输入层神经元个数和输出层神经元个数，==所以需要规定输入的feature的大小。 因此，固定长度的约束仅限于全连接层。以下图为例说明：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514205340.png)

作为全连接层，**如果输入的x维数不等，那么参数w肯定也会不同**，因此，全连接层是必须确定输入，输出个数的。

#### SPP-net 的网络结构

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514210031.png)

SPP-net 最大的亮点便是在传统CNN上加入空间金字塔层(spatial pyramid pooling)，空间金字塔池化的最大做那个用便是，将输入的任意尺度 `feature maps` 组合成特定维度的输出，这个组合可以是不同大小的拼凑，如同拼凑七巧板般。

#### 什么是空间金字塔池化？

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514210358.png)

**黑色图片代表卷积之后的特征图，接着我们以不同大小的块来提取特征**，分别是$4\times4，2\times2，1\times1$，<font color=red>将这三张网格放到下面这张特征图上，就可以得到16+4+1=21种不同的块(Spatial bins)</font>，我们从这21个块中，每个块提取出一个特征，这样刚好就是我们要提取的21维特征向量。**这种以不同的大小格子的组合方式来池化的过程就是空间金字塔池化（SPP）**。比如，要进行空间金字塔最大池化，其实就是从这21个图片块中，分别计算每个块的最大值，从而得到一个输出单元，最终得到一个21维特征的输出。

从整体过程来看，就是如下图所示：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514210632.png)

输出向量大小为Mk，M=#bins， k=#filters，作为全连接层的输入。

​         例如上图，所以Conv5计算出的feature map也是任意大小的，现在经过SPP之后，就可以变成固定大小的输出了，以上图为例，一共可以输出$(16+4+1)\times256$的特征。

总结而言，当网络输入的是一张任意大小的图片，这个时候我们可以一直进行卷积、池化，直到网络的倒数几层的时候，也就是我们即将与全连接层连接的时候，就要使用金字塔池化，使得任意大小的特征图都能够转换成固定大小的特征向量，这就是空间金字塔池化的意义（多尺度特征提取出固定大小的特征向量）。

#### SPP-NET处理流程

- 首先通过选择性搜索，对待检测的图片进行搜索出2000个候选窗口。这一步和R-CNN一样。
-  特征提取阶段。这一步就是和R-CNN最大的区别了，这一步骤的具体操作如下：把整张待检测的图片，输入CNN中，进行一次性特征提取，得到feature maps，然后在feature maps中找到各个候选框的区域，再对各个候选框采用金字塔空间池化，提取出固定长度的特征向量。而R-CNN输入的是每个候选框，然后在进入CNN，因为SPP-Net只需要一次对整张图片进行特征提取，速度会大大提升。

- 采用SVM算法进行特征向量分类识别。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514205746.png)

`SPP Net` 的出现是如同一道惊雷，**不仅减少了计算冗余，更重要的是打破了固定尺寸输入这一束缚**。

[SPP-NET详解](https://blog.csdn.net/v1_vivian/article/details/73275259)

### Fast R-CNN

之所以提出[Fast R-CNN](https://arxiv.org/abs/1504.08083)，主要是因为R-CNN存在以下几个问题：

- 1、训练分多步。我们知道R-CNN的训练先要fine tuning一个预训练的网络，然后针对每个类别都训练一个SVM分类器，最后还要用regressors对bounding-box进行回归，另外region proposal也要单独用selective search的方式获得，步骤比较繁琐。
- 2、时间和内存消耗比较大。在训练SVM和回归的时候需要用网络训练的特征作为输入，特征保存在磁盘上再读入的时间消耗还是比较大的。
- 3、测试的时候也比较慢，每张图片的每个region proposal都要做卷积，重复操作太多

fast R-CNN的流程图如下，网络有两个输入：**图像和对应的region proposal**。其中region proposal由selective search方法得到，没有表示在流程图中。对每个类别都训练一个回归器，且只有非背景的region proposal才需要进行回归。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514212142.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514212251.png)

通过上面R-CNN 和 Fast R-CNN 比较可以看出，原来的 `R-CNN` 是先对候选框区域进行分类，判断有没有物体，如果有则对 `Bounding Box` 进行精修 `回归` ，这是一个串联式的任务。而在Fast R-CNN中，作者 rbg 就将原有结构改成并行——也就是在分类的同时，对 `Bbox` 进行回归。这一改变将 `Bbox` 和 `Clf` 的 `loss` 结合起来变成一个 `Loss` 一起训练，并吸纳了 `SPP Net` 的优点，最终不仅加快了预测的速度，而且提高了精度。

#### Fast RCNN主要有3个改进

- 1、卷积不再是对每个region proposal进行，而是直接对整张图像，这样减少了很多重复计算。原来RCNN是对每个region proposal分别做卷积，因为一张图像中有2000左右的region proposal，肯定相互之间的重叠率很高，因此产生重复计算。

- 2、用ROI pooling进行特征的尺寸变换，因为全连接层的输入要求尺寸大小一样，因此不能直接把region proposal作为输入。

- 3、将regressor放进网络一起训练，每个类别对应一个regressor，同时用softmax代替原来的SVM分类器。

训练时网络结构：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514212850.png)

测试时网络结构：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514212945.png)

总的来说，[Fast-RCNN](https://blog.csdn.net/u014380165/article/details/72851319)将R-CNN的串行网络结构改为并行结构，并且利用SPP-net中提出的空间金字塔池化直接输入整个图像，因此效果比较好。



### Faster R-CNN

在 [Faster R-CNN](https://arxiv.org/abs/1506.01497) 前，我们生产候选区域都是用的一系列启发式算法，基于 `Low Level` 特征生成区域。这样就有两个问题：

**第一个问题** 是生成区域的靠谱程度随缘，而 `两刀流` 算法正是依靠生成区域的靠谱程度——生成大量无效区域则会造成算力的浪费、少生成区域则会漏检；

**第二个问题** 是生成候选区域的算法是在 CPU 上运行的，而我们的训练在 GPU 上面，跨结构交互必定会有损效率。

那么怎么解决这两个问题呢？

Faster R-CNN作者任少卿等人提出了一个 `Region Proposal Networks` 的概念，利用神经网络自己学习去生成候选区域。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514213734.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514213837.png)

这种生成方法同时解决了上述的两个问题，神经网络可以学到更加高层、语义、抽象的特征，生成的候选区域的可靠程度大大提高；可以从上图看出 `RPNs` 和 `RoI Pooling` 共用前面的卷积神经网络——将 `RPNs` 嵌入原有网络，原有网络和 `RPNs` 一起预测，大大地减少了参数量和预测时间。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190514214153.png)

在 `RPNs` 中引入了 `anchor` 的概念，`feature map` 中每个滑窗位置都会生成 `k` 个 `anchors`，然后判断 `anchor` 覆盖的图像是**前景**还是**背景**，同时回归 `Bbox` 的精细位置，预测的 `Bbox` 更加精确。

#### Faster-RCNN 步骤

- <font color=blue> Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。</font>
- <font color=blue>Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于foreground或者background，再利用bounding box regression修正anchors获得精确的proposals。</font>
- <font color=blue>Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。</font>
- <font color=blue>Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。</font>

总的来说：[Faster-RCNN](https://blog.csdn.net/u011746554/article/details/74999010) 将 `两刀流` 的两刀并入同一个网络，这一操作足够载入史册了。 

### R-FCN

[R-FCN](https://blog.csdn.net/baidu_32173921/article/details/71741970)基于共享计算的思想，**由于分类需要特征具有平移不变性，检测则要求对目标的平移做出准确响应。**现在的大部分CNN在分类上可以做的很好，但用在检测上效果不佳。SPP，Faster R-CNN类的方法在[ROI pooling](https://blog.csdn.net/lanran2/article/details/60143861)前都是卷积，是具备平移不变性的，**但一旦插入ROI pooling之后，后面的网络结构就不再具备平移不变性了。**==因此，R-FCN提出来的position sensitive score map这个概念是能把目标的位置信息融合进ROI pooling。== R-FCN整体结构如下：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518124006.png)

对于region-based的检测方法，**以Faster R-CNN为例，实际上是分成了几个subnetwork，**

- **第一个用来在整张图上做比较耗时的conv，这些操作与region无关，是计算共享的。
- ==第二个subnetwork是用来产生候选的boundingbox（如RPN），==
- 第三个subnetwork用来分类或进一步对box进行regression（如Fast RCNN），这个subnetwork和region是有关系的，必须每个region单独跑网络，衔接在这个subnetwork和前两个subnetwork中间的就是ROI pooling。

**我们希望的是，耗时的卷积都尽量移到前面共享的subnetwork上。**因此，和Faster RCNN中用的ResNet（前91层共享，插入ROI pooling，后10层不共享）策略不同，==R-FCN把所有的101层都放在了前面共享的subnetwork。最后用来prediction的卷积只有1层，大大减少了计算量。== 

#### position-sensitive score map

R-FCN会在共享卷积层的最后再接上一层卷积层，而该卷积层就是“位置敏感得分图position-sensitive score map”，该score map是什么意义呢？首先它就是一层卷积层，它的height和width和共享卷积层的一样，但是它的$channels= k^{2}(C+1)$ ，如下图所示

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518130822.png)

假设是人这个类别，那么其有 $k^{2}$个score maps，每一个score map表示“原图image中的哪些位置含有人的某个一个部位”，而该score map会在含有“该score map对应的人体的某个部位”的位置有“高响应值”，换句话说**每一个score map都是用来“描述人体的其中一个部位出现在该score map的何处，而在出现的地方就有高响应值”。** 

[R-FCN](https://zhuanlan.zhihu.com/p/30867916)

[R-FCN详解](https://blog.csdn.net/WZZ18191171661/article/details/79481135)

### Mask R-CNN

我们纵观发展历史，发现 `SPP Net` 升级为 `Fast R-CNN` 时结合了两个 `loss` ，也就是说网络输入了两种信息去训练，结果精度大大提高了。何恺明他们就思考着再加一个信息输入，即图像的 `Mask` ，信息变多之后会不会有提升呢？于是[Mask R-CNN](https://arxiv.org/abs/1703.06870)在Faster R-CNN增加了[掩膜(Mask)](https://blog.csdn.net/jinxiaonian11/article/details/53467437)的概念，，不仅可以做「目标检测」还可以同时做「语义分割」，将两个计算机视觉基本任务融入一个框架。没有使用什么 trick ，性能却有了较为明显的提升，[Mask R-CNN](https://zhuanlan.zhihu.com/p/37998710)如下所示

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518133521.png)

$Mask R-CNN = ResNet-FPN+Fast RCNN+Mask$ 

### R-CNN系列总结

**从平台上讲：**

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518133942.png)

一开始的跨平台，到最后的统一到 GPU 内，效率低到效率高。

**从结构上讲：**

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518134102.png)

一开始的串行到并行，从单一信息流到三条信息流。

从最开始 50s 一张图片的到最后 200ms 一张图片，甚至可以达到 6 FPS 的高精度识别。

## YOLO 系的一刀流

一刀流的想法就比较暴力，**给定一张图像，使用回归的方式输出这个目标的边框和类别。**一刀流最核心的还是利用了分类器优秀的分类效果，首先给出一个大致的范围（最开始就是全图）进行分类，==然后不断迭代这个范围直到一个精细的位置，如下图从蓝色的框框到红色的框框。==

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518164056.png)

这就是一刀流回归的思想，这样做的优点就是快，但是会有许多漏检。

### YOLO v1

[YOLO v1](https://arxiv.org/abs/1506.02640) 将目标检测问题转换为直接从图像中提取bounding boxes和类别概率的单个回归问题，只需一眼（you only look once，YOLO）即可检测目标类别和位置。**YOLO将目标区域预测和目标类别预测整合于单个神经网络模型中，实现在准确率较高的情况下快速目标检测与识别,能够很好的处理实时环境下的目标检测应用。**YOLO处理流程如下：

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518172127.png)

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518165045.png)

首先将图片 `Resize` 到固定尺寸，然后通过一套卷积神经网络，最后接上 `FC` 直接输出结果，这就他们整个网络的基本结构。

更具体地做法，是将输入图片划分成一个 `SxS` 的网格，每个网格负责检测网格里面的物体，并输出 `Bbox Info` 和 `置信度`。这里的置信度指的是 `该网格内含有什么物体` 和 `预测这个物体的准确度`。

更具体的是如下定义：
$$
Pr(Class_{i}|Object) * Pr(Object) * IOU^{truth}_{pred} = Pr(Class_{i}) * IOU^{truth}_{pred}
$$
**可以看出，当框中没有物体的时候，整个置信度都会变为 0 。**

这个想法其实就是一个简单的分而治之想法，将图片卷积后提取的特征图分为 `SxS` 块，然后利用优秀的分类模型对每一块进行分类，将每个网格处理完使用 `NMS` [（非极大值抑制）](https://zhuanlan.zhihu.com/p/37489043)的算法去除重叠的框，最后得到我们的结果。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518170457.png)

#### YOLO优缺点

- 非常快。YOLO预测流程简单，速度很快。基础版在Titan X GPU上可以达到45帧/s； 快速版可以达到150帧/s。因此，YOLO可以实现实时检测。

- YOLO采用全图信息来进行预测。与滑动窗口方法和region proposal-based方法不同，YOLO在训练和预测过程中可以利用全图信息。Fast R-CNN检测方法会错误的将背景中的斑块检测为目标，原因在于Fast R-CNN在检测中无法看到全局图像。相对于Fast R-CNN，YOLO背景预测错误率低一半。

- YOLO可以学习到目标的概括信息（generalizable representation），具有一定普适性。YOLO比其它目标检测方法（DPM和R-CNN）准确率高很多。

- YOLO的准确率没有最好的检测系统准确率高。YOLO可以快速识别图像中的目标，**但是准确定位目标（特别是小目标）有点困难。（定位误差大）**
- 位置精确性差，对于小目标物体以及物体比较密集的也检测不好，比如一群小鸟。 YOLO虽然可以降低将背景检测为物体的概率，但同时导致召回率较低。

[YOLO论文翻译](http://noahsnail.com/2017/08/02/2017-08-02-YOLO%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

### SSD

**YOLO 将目标检测问题转换为直接从图像中提取bounding boxes和类别概率的单个回归问题,这**样做的确非常快，但是问题就在于这个框有点大，就会变得粗糙——小物体就容易从这个大网中漏出去，因此对小物体的检测效果不好。

所以 SSD 就在 YOLO 的基础上添加了 Faster R-CNN 的 Anchor 概念**，并融合不同卷积层的特征做出预测。**

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518182515.png)

我们从上图就可以很明显的看出这是 `YOLO 分治网络` 和 `Faster R-CNN Anchor` 的融合，这就大大提高了对小物体的检测。这里作者做实验也提到和 `Faster R-CNN` 一样的结果，这个 `Anchor`的数量和形状会对性能造成较大的影响。

除此之外，由于这个 `Anchor` 是规整形状的，但是有些物体的摆放位置是千奇百怪的，所以没有 `数据增强` 前的效果比增强后的效果差 7 个百分点。直观点理解，做轻微地角度扭曲让 `Anchor`背后的神经元“看到”更多的信息。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190518184222.png)

还有一个重大的进步是结合了不同尺寸大小 Feature Maps 所提取的特征，然后进行预测。这是 FPN 网络提出前的第一次做 Feature Pyramid 的尝试，这个特征图金字塔结合了不同层的信息，从而结合了不同 `尺寸` 和 `大小` 的特征信息。

这个尝试就大大地提高了识别的精度，且高分辨率（尺寸大）的 Feature Map 中含有更多小物体的信息，也是因为这个原因 [SSD](https://zhuanlan.zhihu.com/p/33544892) 能够较好的识别小物体。

**除此之外，和 YOLO 最大的区别是，SSD 没有接 FC 减少了大量的参数量、提高了速度。**

### YOLO9000

到了 SSD ，回归方法的目标检测应该一统天下了，但是 YOLO 的作者不服气，升级做了一个 YOLO9000 ——号称可以同时识别 9000 类物体的实时监测算法。

讲道理，YOLO9000 更像是 SSD 加了一些 Trick ，而并没有什么本质上的进步：

- Batch Normalization
- High resolution classifier 448*448 pretrain
- Convolution with anchor boxes
- Dimension clusters
- Multi-Scale Training every 10 batch {320，…..608}
- Direct location prediction
- Fine-Grained Features

加了 BN 层，扩大输入维度，使用了 `Anchor`，训练的时候数据增强…

所以强是强，但没啥新意，SSD 和 YOLO9000 可以归为一类。

### 小结 

回顾过去，从 YOLO 到 SSD ，人们兼收并蓄把不同思想融合起来。

YOLO 使用了分治思想，将输入图片分为 `SxS` 的网格，不同网格用性能优良的分类器去分类。
SSD 将 YOLO 和 Anchor 思想融合起来，并创新使用 Feature Pyramid 结构。

但是 `Resize` 输入，必定会损失许多的信息和一定的精度，这也许是一刀流快的原因。

无论如何，YOLO 和 SSD 这两篇论文都是让人不得不赞叹他们想法的精巧，让人受益良多。

### DSSD

## **总结 Summary**

在「目标检测」中有两个指标：`快（Fast）` 和 `准（Accurate）`。

一刀流代表的是快，但是最后在快和准中找到了平衡，第一是快，第二是准。
两刀流代表的是准，虽然没有那么快但是也有 6 FPS 可接受的程度，第一是准，第二是快。

两类算法都有其适用的范围，比如说实时快速动作捕捉，一刀流更胜一筹；复杂、多物体重叠，两刀流当仁不让。没有不好的算法，只有合适的使用场景。

## 参考文献

[FPN+DSSD](https://zhuanlan.zhihu.com/p/26743074)

[YOLOv1-v3](https://www.imooc.com/article/36391)

