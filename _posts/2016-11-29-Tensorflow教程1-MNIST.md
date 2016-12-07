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

<p>其中W代表权重，b代表bias,softmax对计算出来的结果再进行一次处理，将其映射到0到9不同的标签上面去</p>

![3](https://www.tensorflow.org/versions/r0.12/images/softmax-regression-vectorequation.png)

 计算公式为：
\\(y=softmax(Wx+b)\\)

### 代码实现
{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import sys

FLAGS = None

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

    x = tf.placeholder(tf.float32,[None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x,W) + b #matmul means matrix multiplication

    #defince the loss and optimizer
    y_ = tf.placeholder(tf.float32,[None,10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#o.5  is learning rate

    sess = tf.InteractiveSession()

    #train
    tf.global_variables_initializer().run()

    for _ in range(1000): # we will run the train step 1000 times
        batch_xs, batch_ys = mnist.train.next_batch(100) # batch size, for each iteration, we only use 100 points of the dataset
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    #test
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help="Directory for storing data")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv=[sys.argv[0]] + unparsed)

{% endhighlight %}     
<p>下面，我们对上面代码中存在疑问的地方进行解释</p>
* placeholder
<p>这个我们可以理解为占位符，后面的feed_dict中我们要给他具体的值，否则会报错，例如上面代码中我们有x和$$y\_$$ 两个placeholder，因此，需要在feed_dict中给出他们的具体值</p>

* matmul     
<p>这个代表矩阵相乘，因为tensorflow需要适应GPU的运算，所以我们不能再使用python的Numpy的适合CPU的运算方法，因为tensorflow定义了一套自己的运算</p>

* Learning rate
<p>代码中的0.5, learning rate就是梯度下降算法中的，每一步向下走的距离，可以参考梯度下降算法</p>

* batch size
<p>代码中的100, batch size是stochastic gradient descent-随机梯度下降算法中特有的，传统的梯度下降一次需要训练非常多的数据，所以运算复杂度非常高，因此我们在训练的时候每次都只是取其中的一部分来进行训练，batch size就是取出来的数据量的大小</p>
