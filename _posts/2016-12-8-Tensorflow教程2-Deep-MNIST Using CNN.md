---
layout: post
comments: true
categories: Tensorflow
---
## Title

* content
{:toc}

### CNN简介
<p>这里对CNN只做非常简单的介绍，具体的细节，后面的博文会讲到。CNN代表的是Convolutional Neuron Network，这里的Convolution指的卷积，主要是用来对图片的进行进行降维，除了卷积，还有一个pooling的操作，作用也是用来将维的，不过两者的原理不同。</p>
<p>其实一开始训练的图片的时候，一般都直接使用我们的一片教程那样的方法，将层数增多，最后的准确率也可以达到98%； 但是后来，学者就开始思考，仔细观察一张图片，你会发现，图片矩阵中很的0，是一个典型的稀疏矩阵，全连接的神经网络，完全将稀疏的部分和稠密的部分一视同仁，显然效率效率不高，因此出来了现在的CNN。</p>


#### Convolution

#### Pooling
<p>一般用到的Pooling, 叫做max pooling，也就是取矩阵中的最大值，例如下图解释说明，过滤器为2×2的矩阵，stride是2，在tensorflow中对应了tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')这句代码，ksize就是filter, 看第二个和第三个参数，第一个参数代表batch的数量，一般为1，第4个参数代表了channel，因为图片是有颜色channel的，一般为3个channel，因为我们这里是灰色的图片，所以这里为1。stride和ksize是一一对应的，代表在每个方向上面的步数，这里的第二个参数2代表了每次在height方向上面的移动距离，第三个参数代表在width方向上面的移动距离。最后我们取出每个映射举证中的最大值！</p>
![pool](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

#### 结构
<p>下面我们要实现的CNN结果如下图：</p>
![1](https://0sioqw-sn3301.files.1drv.com/y3pEFZqx1USHARK5kZPIdgagvFhUhT0ThBIzF3jzrHCG9gMm76I6XRgN865FAYDKQq1-Mw74fuvuYzwC-9w2g7QWHE3arombd0pJPOGD6T-gRYhn3EZM-Px65Xujc9j2C-EBhhcWcgRR0vWG9o9f4nv6KossTqjgLsySbLZ0nMCvW8/2016-12-08_142414.png?psid=1)
<p>首先，convolution和pool一般是先后进行，这里我们总共进行了两次convolution和两次pooling</p>
### 代码实现
<p>下面看看具体的代码实现</p>
{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import sys

FLAGS = None


def weigt_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#Computes a 2-D convolution given 4-D `input` and `filter` tensors#

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

    x = tf.placeholder(tf.float32,[None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x,W) + b #matmul means matrix multiplication

    #defince the loss and optimizer
    y_ = tf.placeholder(tf.float32,[None,10])


    # first layer
    W_conv1 = weigt_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x,[-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2_2(h_conv1)

    # seconde layer
    W_conv2 = weigt_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2_2(h_conv2)

    # densely connected layer
    W_fc1 = weigt_variable(([7*7*64, 1024]))
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weigt_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    # train
    sess.run(tf.global_variables_initializer())



    for i in range(20000):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0],y_: batch[1], keep_prob: 1.0 })
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help="Directory for storing data")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv=[sys.argv[0]] + unparsed)

{% endhighlight %}
<p>我的输出结果为 1 ！！！百分之百的准确率，也是没谁了，哈哈。</p>



### Reference  

1. https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html
2. https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
3. http://neuralnetworksanddeeplearning.com/chap6.html
4. http://cs.stackexchange.com/questions/49658/convolutional-neural-network-example-in-tensorflow
5. http://www.slideshare.net/ssuser06e0c5/explanation-on-tensorflow-example-deep-mnist-for-expert
