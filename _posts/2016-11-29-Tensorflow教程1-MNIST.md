---
layout: post
comments: true
categories: Tensorflow
---

* content
{:toc}

## Tensorflow教程-1-MNIST

该教程大部分的材料来自tensorflow官网[tensorflow](https://www.tensorflow.org).

### MNIST数据描述

MNIST数据是一个非常经典的机器学习的数据集，它是一个图片数据集，每张有一个手写的阿拉伯数字从0到9，该数据集地址为[MNIST](http://yann.lecun.com/exdb/mnist/), 在Python当中，我们可以使用代码来导入数据，代码如下：
{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
{% endhighlight %} 

MNIST数据有三个部分，训练数据（55000条 mnist.train）, 测试数据(10000条 mnist.test)以及验证数据（5000条 mnist.validation)，这个样的数量分割其实是非常重要的。在做模型训练和测试的时候，一般会将数据按照一定的比例（比如8-2法则）分割。
MNIST的每张图片有28×28个像素，相对真实的图片数据这个已经很简化了，看图片示例：
![1](https://www.tensorflow.org/versions/r0.12/images/MNIST-Matrix.png)
> mnist.train.images 类型为tensor数据维度为[55000,28*28]; mnist.train.labels的维度为[55000,10]，这里的10意思：假如这个数字代表9，则为[0，0，0，0，0，0，0，0，0，1]

### Softmax回归
如果不了解softmax，可以先回顾一下sigmoid，sigmoid用在分类结果只有2个的情况下，softmax用在分类结果有很多种的情况下，详细参见[softmax](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92),下图给出了整个神经网络的结构：
![2](https://www.tensorflow.org/versions/r0.12/images/softmax-regression-scalargraph.png)
> 其中W代表权重，b代表bias,softmax对计算出来的结果再进行一次处理，将其映射到0到9不同的标签上面去

![3](https://www.tensorflow.org/versions/r0.12/images/softmax-regression-vectorequation.png)

 计算公式为：
\\(y=softmax(Wx+b)\\)
