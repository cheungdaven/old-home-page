---
layout: post
comments: true
categories: diary
---
## Title
###MNIST数据描述
MNIST数据是一个非常经典的机器学习的数据集，该数据集地址为[MNIST](http://yann.lecun.com/exdb/mnist/), 在Python当中，我们可以使用代码来导入数据，代码如下：
{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
{% endhighlight %} 
