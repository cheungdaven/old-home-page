---
layout: post
comments: true
categories: Tensorflow
---


## Tensorflow教程-1-MNIST
### MNIST数据描述
MNIST数据是一个非常经典的机器学习的数据集，它是一个图片数据集，每张有一个手写的阿拉伯数字从0到9，该数据集地址为[MNIST](http://yann.lecun.com/exdb/mnist/), 在Python当中，我们可以使用代码来导入数据，代码如下：
{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
{% endhighlight %} 

MNIST数据有三个部分，训练数据（55000条 mnist.train）, 测试数据(10000条 mnist.test)以及验证数据（5000条 mnist.validation)，这个样的数量分割其实是非常重要的。在做模型训练和测试的时候，一般会将数据按照一定的比例（比如8-2法则）分割。
MNIST的每张图片有28×28个像素，相对真实的图片数据这个已经很简化了，看图片示例：
![1](https://www.tensorflow.org/versions/r0.12/images/MNIST-Matrix.png)

### MNIST数据描述
