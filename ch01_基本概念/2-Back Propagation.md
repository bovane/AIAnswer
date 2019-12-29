[TOC]

# What is Back Propagation ?

反向传播算法$BP$是“误差反向传播”的简称，是一种与最优化方法（如梯度下降法结合使用的，用来训练人工神经网络的常见方法。该方法对网络中所有权重计算**损失函数的梯度**。这个梯度会反馈给最优化方法，**用来更新权值以最小化损失函数。**

## 算法

算法具体描述参见参考文献[1]、[2]

![1551362631393](C:\Users\bovan\AppData\Roaming\Typora\typora-user-images\1551362631393.png)

![1551362836012](C:\Users\bovan\AppData\Roaming\Typora\typora-user-images\1551362836012.png)

## 实现BP算法

```python
# BP Demo
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

#基于seed产生随机数
rdm = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rdm.rand(32,2)
#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0 
#作为输入数据集的标签（正确答案） 
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

#1定义神经网络的输入、参数和输出,定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2定义损失函数及反向传播方法。
loss_mse = tf.reduce_mean(tf.square(y-y_)) 
#均方误差MSE损失函数
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#随机梯度下降算法训练参数

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。

    # 训练模型。
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            #每训练500个steps打印训练误差
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
```

## 参考文献

- [实例演算BP过程](https://zhuanlan.zhihu.com/p/32819991)
- [维基百科](https://zh.wikipedia.org/zh-hans/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95) 
- [BP实现DEMO](https://zhuanlan.zhihu.com/p/35014526)
- [BP实现DEMO2](https://blog.csdn.net/weixin_39198406/article/details/82183854)





