[TOC]

# Padding

关于卷积神经网络中的Padding理解，当我们对图像进行卷积操作时，会产生两个问题：

- 卷积之后的图像越来越小
- 输入矩阵左**边缘像素**(绿色位置)只被计算过一次，而中间像素被卷积计算多次，意味着丢失图像角落信息。

为了解决这两个问题，我们可以对输入图像进行Padding，即像素填充。

![](https://raw.githubusercontent.com/bovane/md_images/master/20190307131502.png)

通过Padding之后，一方面可以增加卷积后的图像大小，另一方面削弱边缘信息丢失的程度。

## 卷积的两种策略

- Valid卷积：不对图像进行任何填充处理
- Same卷积：对图像进行Padding处理，使得输入和输出图像大小一致 $p=\frac{f-1}{2}$，$f$为卷积核大小

通常计算机视觉使用的[滤波器](https://www.baidu.com/s?wd=%E6%BB%A4%E6%B3%A2%E5%99%A8&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)矩阵f都是奇数。（可能是由于奇数f使p填充时对称；奇数f使滤波器矩阵有中心像素点，计算机视觉中方便指出滤波矩阵位置）

## Tensorflow实现

```python
conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
```

## 参考文献

[卷积层计算细节](https://zhuanlan.zhihu.com/p/29119239)







